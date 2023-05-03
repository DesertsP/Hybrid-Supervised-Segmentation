import torch
import torch.nn as nn
from utils import build_module
from models.mods.ops import set_requires_grad, resize_as, update_model_moving_average
from kornia import augmentation as augs
from datasets.augmentation import generate_unsup_data
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


class MeanTeacherLearner(nn.Module):
    def __init__(self, num_classes, num_samples, indicator_initialization=0.0, ema_decay=0.99,
                 network_cfg: dict = None):
        super().__init__()
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
        indicators = torch.ones(num_samples) * indicator_initialization
        self.register_buffer('indicators', indicators)
        # unbiased gradient cache
        unbiased_grad = torch.zeros(self.online_net.classification_loss.out_channels * self.online_net.classification_loss.in_channels)
        self.register_buffer('unbiased_grad', unbiased_grad)
        self.proxy_classifier = None
        self.proxy_criterion = extend(nn.CrossEntropyLoss(ignore_index=255, reduction='none'))
        self.hessian_type = 'identity'

    @singleton('proxy_classifier')
    def get_proxy_classifier(self):
        proxy_classifier = copy.deepcopy(self.online_net.classification_loss)
        proxy_classifier = extend(proxy_classifier)
        return proxy_classifier

    def update_proxy_classifier(self):
        classifier = self.get_proxy_classifier()
        for t_p, o_p in zip(classifier.parameters(), self.online_net.classification_loss.parameters()):
            t_p.data.copy_(o_p.data)
        return classifier

    @torch.no_grad()
    def extract_feature(self, image):
        x = self.normalize(image)
        x = self.online_net.backbone(x)
        x = self.online_net.decoder(x)
        x = self.online_net.projector(x)
        return x

    def grad_classifier(self, image, target):
        """
        Compute holistic gradients w.r.t classifier weights.
        Returns:
        """
        x = self.extract_feature(image)
        classifier = self.update_proxy_classifier()
        pred = classifier(x)
        loss = self.proxy_criterion(resize_as(pred, target), target).mean()
        loss.backward()
        grad = classifier.weight.grad.detach().clone()
        classifier.zero_grad()
        grad = grad.reshape(-1)
        return grad

    def grad_per_sample_classifier(self, image, target):
        """
        Compute per-sample gradients w.r.t classifier weights.
        For coarse/weakly labeled data point, we can use the coarse labels to compute loss.
        For unlabeled data, we can consider the regularization/pseudo-label loss.
        Returns:
        """
        x = self.extract_feature(image)
        classifier = self.update_proxy_classifier()
        pred = classifier(x)
        loss = self.proxy_criterion(resize_as(pred, target), target).sum()
        with backpack(BatchGrad()):
            loss.backward()
        grad = classifier.weight.grad_batch.detach().clone()
        classifier.zero_grad()
        grad = einops.rearrange(grad, 'n o i 1 1-> n (o i)')
        return grad

    def compute_influence(self, image, target):
        unbiased_grad = self.unbiased_grad.unsqueeze(0)     # (i*o,) -> (1, i*o)
        grad_per_sample = self.grad_per_sample_classifier(image, target)

        if self.hessian_type == 'identity':
            influence = -unbiased_grad.matmul(grad_per_sample.permute(1, 0))
        else:
            # compute / estimate inverse Hessian
            h_ = self.hessian_classifier(image, target)
            h_inv = h_.inverse()
            influence = -unbiased_grad.matmul(h_inv).matmul(grad_per_sample.permute(1, 0))
        return influence.reshape(-1)

    @torch.no_grad()
    def sync_indicators(self, indicator_grad, index):
        process_group = dist.group.WORLD
        world_size = dist.get_world_size(process_group)
        if world_size <= 1:
            return
        index_all = torch.empty(world_size, index.size(0), dtype=index.dtype, device=index.device)
        indicator_grad_all = torch.empty(world_size, indicator_grad.size(0),
                                         dtype=indicator_grad.dtype, device=indicator_grad.device)
        index_l = list(index_all.unbind(0))
        indicator_grad_l = list(indicator_grad_all.unbind(0))
        index_gather = dist.all_gather(index_l, index, process_group, async_op=False)
        indicator_gather = dist.all_gather(indicator_grad_l, indicator_grad, process_group, async_op=False)
        # index_gather.wait()
        # indicator_gather.wait()
        for idx, val in zip(index_l, indicator_grad_l):
            self.indicators.grad[idx] = val

    @torch.no_grad()
    def sync_unbiased_grad(self):
        process_group = dist.group.WORLD
        world_size = dist.get_world_size(process_group)
        unbiased_grad_all = torch.empty(world_size, self.unbiased_grad.size(0),
                                        dtype=self.unbiased_grad.dtype, device=self.unbiased_grad.device)
        unbiased_grad_l = list(unbiased_grad_all.unbind(0))
        unbiased_grad_gather = dist.all_gather(unbiased_grad_l, self.unbiased_grad,
                                               process_group, async_op=False)
        # unbiased_grad_gather.wait()
        _mean = torch.stack(unbiased_grad_l, dim=0).mean(0)
        self.unbiased_grad.data.copy_(_mean.data)

    def compute_unbiased_grad(self, image, target, ema_decay=0.9, sync=True):
        grads = self.grad_classifier(image, target)
        if ema_decay is None:
            self.unbiased_grad.data = grads.data
        else:
            self.unbiased_grad.data = ema_decay * self.unbiased_grad.data + (1 - ema_decay) * grads.data
        if sync:
            self.sync_unbiased_grad()
        return grads

    def compute_indicator_grad(self, image, target=None, index=None, sync=True):
        if target is None:
            _, target, _ = self.forward_pseudo_label(image, augment='none')
        influence = self.compute_influence(image, target)
        if self.indicators.grad is None:
            self.indicators.grad = torch.zeros_like(self.indicators)
        assert index is not None
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
    def forward_pseudo_label(self, image, keep_ratio=80, augment='cutmix'):
        # generate pseudo labels first
        image = self.normalize(image)
        # self.target_net.train()
        pred_u_teacher = self.target_net(image)["pred"]
        pred_u_teacher = resize_as(pred_u_teacher, image)
        pred_u_teacher = pred_u_teacher.softmax(dim=1)
        logits_u, label_u = torch.max(pred_u_teacher, dim=1)
        entropy = -torch.sum(pred_u_teacher * torch.log(pred_u_teacher + 1e-10), dim=1)
        # apply strong data augmentation: cutout, cutmix, or classmix
        use_mix = False
        if augment and augment != 'none' and np.random.uniform(0, 1) < 0.5:
            use_mix = True
            image, label_u, entropy = generate_unsup_data(image, label_u.clone(), entropy.clone(), mode=augment)
        # filter out high entropy pixels
        with torch.no_grad():
            thresh = np.percentile(
                entropy.flatten().detach().cpu().numpy(), keep_ratio
            )
            thresh_mask = entropy.ge(thresh).bool() * (label_u != 255).bool()
            label_u[thresh_mask] = 255
        return image, label_u, use_mix

    def forward_train(self, image, target, image_u, index_u):
        image_u, target_u, use_mix = self.forward_pseudo_label(image_u.clone())
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
        indicator = self.indicators[index_u]
        if use_mix:
            indicator = (indicator + indicator.roll(-1)) / 2.0
        indicator = indicator[:, None, None]
        pseudo_loss = self.criterion(resize_as(pred_u, target_u), target_u.clone())
        pseudo_loss *= indicator
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
                       self.online_net.classification_loss
                       ]
        for module in backbone:
            for p in module.parameters():
                groups[0].append(p)
        for module in newly_added:
            for p in module.parameters():
                groups[1].append(p)
        assert len(list(self.parameters())) == 2 * sum([len(g) for g in groups])
        return groups
