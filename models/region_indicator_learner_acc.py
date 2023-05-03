import torch
import torch.nn as nn
from utils import build_module
from models.mods.ops import set_requires_grad, resize_as, update_model_moving_average
from kornia import augmentation as augs
from datasets.augmentation import generate_cutout_mask, generate_class_mask
import numpy as np
from torch.nn import functional as F
from backpack import extend, backpack
from backpack.extensions import BatchGrad
from functools import wraps
import copy
import einops
import torch.distributed as dist


def singleton(cache_key):
    def inner_fn(fn):
        @wraps(fn)
        def wrapper(self, *args, **kwargs):
            instance = getattr(self, cache_key)
            if instance is not None:
                return instance
            instance = fn(self, *args, **kwargs)
            setattr(self, cache_key, instance)
            return instance

        return wrapper

    return inner_fn

def label_onehot(inputs, num_classes):
    batch_size, im_h, im_w = inputs.shape
    outputs = torch.zeros((num_classes, batch_size, im_h, im_w)).to(inputs.device)

    inputs_temp = inputs.clone()
    inputs_temp[inputs == 255] = 0
    outputs.scatter_(0, inputs_temp.unsqueeze(1), 1.0)
    outputs[:, inputs == 255] = 0

    return outputs.permute(1, 0, 2, 3)

def pseudo_segmentation_loss(predict, target, percent, entropy):
    batch_size, num_class, h, w = predict.shape

    with torch.no_grad():
        # drop pixels with high entropy
        # entropy = -torch.sum(prob * torch.log(prob + 1e-10), dim=1)

        thresh = np.percentile(
            entropy[target != 255].detach().cpu().numpy().flatten(), percent
        )
        thresh_mask = entropy.ge(thresh).bool() * (target != 255).bool()

        target[thresh_mask] = 255
        weight = batch_size * h * w / torch.sum(target != 255)

    loss = weight * F.cross_entropy(predict, target, ignore_index=255)

    return loss


def weighted_gather(x, p):
    x = einops.rearrange(x, 'b k h w -> b k (h w)')
    p = einops.rearrange(p, 'b c h w -> b c (h w)')
    p = p / (p.sum(dim=-1, keepdim=True) + 1e-6)
    mean_ = p.matmul(x.permute(0, 2, 1))  # (b, c, n)(b, n, k)->(b, c, k)
    return mean_


def generate_masks(image, target=None, mode="cutout"):
    batch_size, _, im_h, im_w = image.shape
    device = image.device
    mask_list = []
    for i in range(batch_size):
        if mode == "cutmix":
            mask_list.append(generate_cutout_mask([im_h, im_w]))
        elif mode == "classmix":
            mask_list.append(generate_class_mask(target[i]))
        else:
            raise ValueError(f'Unexpected mode: {mode}')
    masks = torch.stack(mask_list, dim=0)       # (b, h, w)
    masks = masks.to(device)
    return masks


class MeanTeacherLearner(nn.Module):
    def __init__(self, num_classes, num_samples, indicator_initialization=0.0, ema_decay=0.99,
                 pseudo_augment='cutmix',
                 pseudo_keep_ratio=80,
                 pseudo_augment_prob=0.5,
                 hyper_criterion_cfg=None,
                 network_cfg: dict = None):
        super().__init__()
        self.pseudo_augment = pseudo_augment
        self.pseudo_augment_prob = pseudo_augment_prob
        self.pseudo_keep_ratio = pseudo_keep_ratio
        self.online_net = build_module(**network_cfg)
        self.target_net = build_module(**network_cfg)
        for t_p, o_p in zip(self.target_net.parameters(), self.online_net.parameters()):
            t_p.data.copy_(o_p.data)
        set_requires_grad(self.target_net, False)
        self.criterion = nn.CrossEntropyLoss(ignore_index=255, reduction='none')

        NORMALIZE = augs.Normalize(mean=torch.tensor((0.485, 0.456, 0.406)),
                                   std=torch.tensor((0.229, 0.224, 0.225)))
        STRONG_AUG = nn.Sequential(
            augs.ColorJitter(0.5, 0.5, 0.5, 0.25, p=0.8),
            augs.RandomGrayscale(p=0.2),
            augs.RandomGaussianBlur((3, 3), (0.1, 2.0), p=0.5),
            augs.RandomSolarize(p=0.1),
            NORMALIZE
        )
        DEFAULT_AUG = NORMALIZE
        self.augment = DEFAULT_AUG
        self.strong_augment = STRONG_AUG
        self.normalize = NORMALIZE
        self.num_classes = num_classes
        self.ema_decay = ema_decay
        # for per-sample weight learning, hack the backprop. to obtain gradients w.r.t each sample
        indicators = torch.ones(num_samples, num_classes) * indicator_initialization
        self.register_buffer('indicators', indicators)
        # unbiased gradient cache
        unbiased_grad = torch.zeros(self.online_net.classifier.out_channels * self.online_net.classifier.in_channels)
        self.register_buffer('unbiased_grad', unbiased_grad)
        self.proxy_classifier = None
        self.proxy_criterion = extend(nn.CrossEntropyLoss(ignore_index=255, reduction='none'))
        if hyper_criterion_cfg is None:
            self.hyper_criterion = nn.CrossEntropyLoss(ignore_index=255)
        else:
            self.hyper_criterion = build_module(**hyper_criterion_cfg)
        self.hessian_type = 'identity'

    @singleton('proxy_classifier')
    def get_proxy_classifier(self):
        proxy_classifier = copy.deepcopy(self.online_net.classifier)
        proxy_classifier = extend(proxy_classifier)
        return proxy_classifier

    def update_proxy_classifier(self, copy_parameters=True):
        classifier = self.get_proxy_classifier()
        if not copy_parameters:
            return classifier
        for t_p, o_p in zip(classifier.parameters(), self.online_net.classifier.parameters()):
            t_p.data.copy_(o_p.data)
        classifier.zero_grad()
        return classifier

    @torch.no_grad()
    def extract_feature(self, image):
        x = self.normalize(image)
        x = self.online_net.backbone(x)
        x = self.online_net.decoder(x)
        x = self.online_net.projector(x)
        return x.detach()

    def grad_classifier(self, image, target, zero_grad=False):
        """
        Compute holistic gradients w.r.t classifier weights.
        Returns:
        """
        x = self.extract_feature(image)
        if zero_grad:
            classifier = self.update_proxy_classifier(copy_parameters=True)
            classifier.zero_grad()
        else:
            classifier = self.update_proxy_classifier(copy_parameters=False)
        pred = classifier(x)
        loss = self.hyper_criterion(resize_as(pred, target), target)
        loss.backward()
        grad = classifier.weight.grad.detach().clone()
        grad = grad.reshape(-1)
        return grad

    def grad_per_region_classifier(self, image, target, copy_parameters=False):
        """
        Compute per-region gradients w.r.t classifier weights.
        For coarse/weakly labeled data point, we can use the coarse labels to compute loss.
        For unlabeled data, we can consider the regularization/pseudo-label loss.
        Returns:
        """
        x = self.extract_feature(image)
        classifier = self.update_proxy_classifier(copy_parameters=copy_parameters)

        b, c, h, w = x.shape
        representations = einops.rearrange(x, 'b c h w -> (b h w) c 1 1')
        pred = classifier(representations)
        pred = einops.rearrange(pred, '(b h w) c 1 1 -> b c h w', b=b, h=h, w=w)

        loss = self.proxy_criterion(resize_as(pred, target), target).sum()
        with backpack(BatchGrad()):
            loss.backward()
        grad = classifier.weight.grad_batch.detach().clone()
        # grad = einops.rearrange(grad, '(b h w) o i 1 1-> b (o i) h w', b=b, h=h, w=w)
        with torch.no_grad():
            grad = einops.rearrange(grad, '(b h w) o i 1 1-> b (o i) h w', b=b, h=h, w=w)
            target_oh = resize_as(label_onehot(target, self.num_classes), grad, mode='nearest', align_corners=None)
            grad_region = weighted_gather(grad, target_oh)     # (b, #classes, k)
        return grad_region

    def compute_influence(self, image, target):
        unbiased_grad = self.unbiased_grad.unsqueeze(0)     # (1,i*o)
        grad_per_region = self.grad_per_region_classifier(image, target)

        if self.hessian_type == 'identity':
            influence = -unbiased_grad.matmul(einops.rearrange(grad_per_region, 'b c k -> k (b c)'))
        else:
            # compute / estimate inverse Hessian
            h_ = self.hessian_classifier(image, target)
            h_inv = h_.inverse()
            influence = -unbiased_grad.matmul(h_inv).matmul(einops.rearrange(grad_per_region, 'b c k -> k (b c)'))
        influence = einops.rearrange(influence, '1 (b c) ->  b c', b=grad_per_region.size(0), c=grad_per_region.size(1))
        return influence

    @torch.no_grad()
    def sync_indicators(self, indicator_grad, index):
        process_group = dist.group.WORLD
        world_size = dist.get_world_size(process_group)
        if world_size <= 1:
            return
        index_all = torch.empty(world_size, *index.size(), dtype=index.dtype, device=index.device)
        indicator_grad_all = torch.empty(world_size, *indicator_grad.size(),
                                         dtype=indicator_grad.dtype, device=indicator_grad.device)
        index_l = list(index_all.unbind(0))
        indicator_grad_l = list(indicator_grad_all.unbind(0))
        index_gather = dist.all_gather(index_l, index, process_group, async_op=False)
        indicator_gather = dist.all_gather(indicator_grad_l, indicator_grad, process_group, async_op=False)
        for idx, val in zip(index_l, indicator_grad_l):
            self.indicators.grad[idx] = val

    @torch.no_grad()
    def sync_unbiased_grad(self):
        process_group = dist.group.WORLD
        world_size = dist.get_world_size(process_group)
        unbiased_grad_all = torch.empty(world_size, self.unbiased_grad.size(0),
                                        dtype=self.unbiased_grad.dtype, device=self.unbiased_grad.device)
        unbiased_grad_l = list(unbiased_grad_all.unbind(0))
        dist.all_gather(unbiased_grad_l, self.unbiased_grad,
                        process_group, async_op=False)
        _mean = torch.stack(unbiased_grad_l, dim=0).mean(0)
        self.unbiased_grad.data.copy_(_mean.data)

    def compute_unbiased_grad(self, image, target, zero_grad=False, sync=False):
        grads = self.grad_classifier(image, target, zero_grad=zero_grad)
        self.unbiased_grad.data = grads.data
        if sync:
            self.sync_unbiased_grad()
        return grads

    def compute_indicator_grad(self, image, target=None, index=None, sync=True):
        assert index is not None
        if target is None:
            _, target, _ = self.forward_pseudo_label(image, index, use_augment=False)
        influence = self.compute_influence(image, target)
        if self.indicators.grad is None:
            self.indicators.grad = torch.zeros_like(self.indicators)
        if sync:
            self.sync_indicators(influence, index)
        else:
            self.indicators.grad[index] = influence
        return influence

    def hessian_classifier(self, image, target):
        """
        TODO: The complexity of calculating the Hessian matrix directly is terrible.
        """
        raise NotImplementedError

    @torch.no_grad()
    def forward_pseudo_label(self, image, index, use_augment=True):
        # generate pseudo labels first
        image = self.normalize(image)
        # self.target_net.train()
        pred_u_teacher = self.target_net(image)["pred"]
        pred_u_teacher = resize_as(pred_u_teacher, image)
        pred_u_teacher = pred_u_teacher.softmax(dim=1)
        logits_u, label_u = torch.max(pred_u_teacher, dim=1)
        entropy = -torch.sum(pred_u_teacher * torch.log(pred_u_teacher + 1e-10), dim=1)
        # apply strong data augmentation: cutmix, or classmix
        indicator = self.indicators[index]      # (b, c)
        indicator_p = label_onehot(label_u, self.num_classes)
        indicator_p = indicator_p * einops.rearrange(indicator, 'b c -> b c 1 1')   # (b,c,h,w) * (b,c)
        indicator_p = indicator_p.sum(dim=1)
        if use_augment and self.pseudo_augment and np.random.uniform(0, 1) < self.pseudo_augment_prob:
            mix_mask = generate_masks(image, label_u, mode=self.pseudo_augment)
            mix_mask_unsqz = mix_mask.unsqueeze(1)
            image = image * mix_mask_unsqz + image.roll(-1, dims=0) * (1 - mix_mask_unsqz)
            label_u = label_u * mix_mask + label_u.roll(-1, dims=0) * (1 - mix_mask)
            label_u = label_u.long()
            entropy = entropy * mix_mask + entropy.roll(-1, dims=0) * (1 - mix_mask)
            indicator_p = indicator_p * mix_mask + indicator_p.roll(-1, dims=0) * (1 - mix_mask)
        # filter out high entropy pixels
        if self.pseudo_keep_ratio < 100:
            thresh = np.percentile(entropy.flatten().detach().cpu().numpy(), self.pseudo_keep_ratio)
            thresh_mask = entropy.ge(thresh).bool() * (label_u != 255).bool()
            label_u[thresh_mask] = 255
        return image, label_u, indicator_p

    def forward_train(self, image, target, image_u, index_u):
        image_u, target_u, indicator_u = self.forward_pseudo_label(image_u.clone(), index_u)
        image = self.augment(image)
        image_u = self.strong_augment(image_u)
        image_all = torch.cat([image, image_u], dim=0)
        outputs = self.online_net(image_all)
        pred_all = outputs['pred']
        num_labeled, num_unlabeled = image.size(0), image_u.size(0)
        pred, pred_u = torch.split(pred_all, [num_labeled, num_unlabeled], dim=0)

        # supervised loss
        sup_loss = self.criterion(resize_as(pred, image), target).mean()

        # unsupervised loss
        pseudo_loss = self.criterion(resize_as(pred_u, target_u), target_u.clone())
        pseudo_loss *= indicator_u
        pseudo_loss = pseudo_loss.mean()
        # rescale
        b, h, w = target_u.shape
        scale_factor = b*h*w / torch.sum(target_u != 255)
        pseudo_loss *= scale_factor

        # moving average
        update_model_moving_average(self.ema_decay, self.target_net, self.online_net)
        return dict(seg_loss=sup_loss, reg_loss=pseudo_loss)

    def forward_test(self, input):
        x = self.normalize(input)
        pred = self.online_net(x)
        return pred

    def forward(self, *args, mode=0, **kwargs):
        if mode == 1:
            assert self.training
            return self.forward_train(*args, **kwargs)
        elif mode == 2:
            return self.compute_unbiased_grad(*args, **kwargs)
        elif mode == 3:
            return self.compute_indicator_grad(*args, **kwargs)
        else:
            return self.forward_test(*args, **kwargs)

    def parameter_groups(self):
        groups = ([], [])
        backbone = [self.online_net.backbone]
        newly_added = [self.online_net.decoder,
                       self.online_net.projector,
                       self.online_net.classifier
                       ]
        for module in backbone:
            for p in module.parameters():
                groups[0].append(p)
        for module in newly_added:
            for p in module.parameters():
                groups[1].append(p)
        assert len(list(self.parameters())) == 2 * sum([len(g) for g in groups])
        return groups
