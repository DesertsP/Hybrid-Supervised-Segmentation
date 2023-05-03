import torch
import torch.nn as nn
from utils import build_module
from models.mods.ops import set_requires_grad, resize_as, update_model_moving_average, copy_parameters,\
    label_onehot, singleton, weighted_gather, generate_mix_masks
import einops
from kornia import augmentation as augs
import numpy as np
from backpack import extend, backpack
from backpack.extensions import BatchGrad, DiagHessian
import copy
import torch.distributed as dist


class IndicatorLearner(nn.Module):
    def __init__(self, num_classes, num_samples, indicator_initialization=0.0, ema_decay=0.99,
                 pseudo_augment='cutmix',
                 pseudo_keep_ratio=100,
                 hyper_criterion_cfg=None,
                 network_cfg: dict = None,
                 hessian_type='identity',
                 ignore_index=255,
                 **kwargs):
        super().__init__()
        self.pseudo_augment = pseudo_augment
        self.pseudo_keep_ratio = pseudo_keep_ratio
        self.online_net = build_module(**network_cfg)
        self.target_net = build_module(**network_cfg)
        for t_p, o_p in zip(self.target_net.parameters(), self.online_net.parameters()):
            t_p.data.copy_(o_p.data)
        set_requires_grad(self.target_net, False)
        self.criterion = nn.CrossEntropyLoss(ignore_index=ignore_index, reduction='none')

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
        self.proxy_criterion = extend(nn.CrossEntropyLoss(ignore_index=ignore_index, reduction='none'))
        if hyper_criterion_cfg is None:
            self.hyper_criterion = nn.CrossEntropyLoss(ignore_index=ignore_index)
        else:
            self.hyper_criterion = build_module(**hyper_criterion_cfg)
        self.hessian_type = hessian_type
        self.ignore_index = ignore_index

    @singleton('proxy_classifier')
    def get_proxy_classifier(self):
        proxy_classifier = copy.deepcopy(self.online_net.classifier)
        proxy_classifier = extend(proxy_classifier)
        return proxy_classifier

    @torch.no_grad()
    def extract_feature(self, image):
        x = self.online_net.backbone(image)
        x = self.online_net.decoder(x)
        x = self.online_net.projector(x)
        return x.detach()

    def grad_classifier(self, image, target, zero_grad=True, sync=True, accumulate_iters=1):
        """
        Compute holistic gradients w.r.t classifier weights.
        Returns:
        """
        x = self.extract_feature(self.strong_augment(image))
        classifier = self.get_proxy_classifier()
        if zero_grad:       # aggregate grad if False
            classifier = copy_parameters(classifier, self.online_net.classifier)
            classifier.zero_grad()
        pred = classifier(x)
        loss = (1 / accumulate_iters) * self.hyper_criterion(resize_as(pred, target), target)
        loss.backward()
        grad = classifier.weight.grad.detach().clone()
        # grad_b = classifier.bias.grad.detach().clone()
        # grad = torch.cat([grad_w.reshape(-1), grad_b])
        grad = grad.reshape(-1)
        if sync:
            grad = self.sync_tensors(grad)
        self.unbiased_grad.data.copy_(grad.data)
        return grad

    def hessian_diagonal_classifier(self, image, target, zero_grad=True):
        """
        NOTE: The complexity of calculating the Hessian matrix directly is terrible.
        """
        x = self.extract_feature(self.strong_augment(image))
        classifier = self.get_proxy_classifier()
        if zero_grad:  # aggregate grad if False
            classifier = copy_parameters(classifier, self.online_net.classifier)
            classifier.zero_grad()
        pred = classifier(x)
        loss = self.proxy_criterion(resize_as(pred, target), target).mean()
        with backpack(DiagHessian()):
            loss.backward()
        diag_hess = classifier.weight.diag_h.detach().clone()
        return diag_hess.reshape(-1)

    def grad_per_region_classifier(self, image, target, update_parameters=False):
        """
        Compute per-region gradients w.r.t classifier weights.
        For coarse/weakly labeled data point, we use the coarse labels to compute loss.
        For unlabeled data, we consider the pseudo-label loss.
        Returns: per-regional gradients
        NOTE:
            BackPACK does not aggregate/accumulate grad like PyTorch.
            Every call to .backward(), inside a with backpack(...):, reset the corresponding field,
            and the fields returned by BackPACK are not affected by zero_grad().
        """
        x = self.extract_feature(self.strong_augment(image))
        classifier = self.get_proxy_classifier()
        if update_parameters:
            copy_parameters(classifier, self.online_net.classifier)
        classifier.zero_grad()

        b, c, h, w = x.shape
        representations = einops.rearrange(x, 'b c h w -> (b h w) c 1 1')
        pred = classifier(representations)
        pred = einops.rearrange(pred, '(b h w) c 1 1 -> b c h w', b=b, h=h, w=w)

        loss = self.proxy_criterion(resize_as(pred, target), target).sum()
        with backpack(BatchGrad()):
            loss.backward()
        grad_w = classifier.weight.grad_batch.detach().clone()
        # grad_b = classifier.bias.grad_batch.detach().clone()
        with torch.no_grad():
            grad_w = einops.rearrange(grad_w, '(b h w) o i 1 1-> b (o i) h w', b=b, h=h, w=w)
            # grad_b = einops.rearrange(grad_b, '(b h w) o -> b o h w', b=b, h=h, w=w)
            # grad = torch.cat([grad_w, grad_b], dim=1)
            grad = grad_w
            target_oh = label_onehot(target, self.num_classes)
            target_oh = resize_as(target_oh.float(), grad, mode='nearest', align_corners=None)
            grad_region = weighted_gather(grad, target_oh)     # (b, #regions, k)
        return grad_region

    def compute_influence(self, image, target, image_unbiased, target_unbiased, eps=1e-8):
        unbiased_grad = self.grad_classifier(image_unbiased, target_unbiased,
                                             zero_grad=True, sync=True, accumulate_iters=1)     # (k,)
        grad_per_region = self.grad_per_region_classifier(image, target, update_parameters=False)

        unbiased_grad = unbiased_grad.unsqueeze(0)
        if self.hessian_type == 'identity':
            influence = -unbiased_grad.matmul(einops.rearrange(grad_per_region, 'b r k -> k (b r)'))
            influence = einops.rearrange(influence, '1 (b r) ->  b r', b=grad_per_region.size(0),
                                         r=grad_per_region.size(1))
        elif self.hessian_type == 'diagonal':
            # estimate inverse Hessian
            image_all = torch.cat([image, image_unbiased], dim=0)
            target_all = torch.cat([target, target_unbiased], dim=0)
            hess = self.hessian_diagonal_classifier(image_all, target_all)
            hess_inv = 1.0 / (hess + eps)
            influence = -(unbiased_grad * hess_inv.unsqueeze(0)).matmul(einops.rearrange(grad_per_region,
                                                                                         'b r k -> k (b r)'))
            influence = einops.rearrange(influence, '1 (b r) ->  b r', b=grad_per_region.size(0),
                                         r=grad_per_region.size(1))
        elif self.hessian_type == 'full':
            # exact hessian
            h_ = self.hessian_classifier(image, target)
            h_inv = h_.inverse()
            influence = -unbiased_grad.matmul(h_inv).matmul(einops.rearrange(grad_per_region, 'b c k -> k (b c)'))
        else:
            raise NotImplementedError
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
        dist.all_gather(index_l, index, process_group, async_op=False)
        dist.all_gather(indicator_grad_l, indicator_grad, process_group, async_op=False)
        index_all = torch.cat(index_l, dim=0)
        indicator_grad_all = torch.cat(indicator_grad_l, dim=0)
        self.indicators.grad[index_all] = indicator_grad_all
        return index_all

    @torch.no_grad()
    def sync_tensors(self, tensor):
        process_group = dist.group.WORLD
        world_size = dist.get_world_size(process_group)
        tensor_all = torch.empty(world_size, *tensor.size(), dtype=tensor.dtype, device=tensor.device)
        tensor_l = list(tensor_all.unbind(0))
        dist.all_gather(tensor_l, tensor,
                        process_group, async_op=False)
        _reduced = torch.stack(tensor_l, dim=0).mean(0)
        tensor.data.copy_(_reduced)
        return tensor

    def compute_indicator_grad(self, image, target, index, image_unbiased, target_unbiased, sync=True):
        assert index is not None
        if target is None:
            target = self.forward_pseudo_label(image)
        influence = self.compute_influence(image, target, image_unbiased, target_unbiased)
        if self.indicators.grad is None:
            self.indicators.grad = torch.zeros_like(self.indicators)
        assert index.size(0) == influence.size(0)
        if sync:
            return self.sync_indicators(influence, index)
        else:
            self.indicators.grad[index] = influence
            return influence

    @torch.no_grad()
    def forward_pseudo_label(self, image):
        # generate pseudo labels first
        image = self.normalize(image)
        # self.target_net.train()
        pred_u_teacher = self.target_net(image)["pred"]
        pred_u_teacher = resize_as(pred_u_teacher, image)
        pred_u_teacher = pred_u_teacher.softmax(dim=1)
        logits_u, label_u = torch.max(pred_u_teacher, dim=1)
        # filter out high entropy pixels
        if self.pseudo_keep_ratio < 100:
            entropy = -torch.sum(pred_u_teacher * torch.log(pred_u_teacher + 1e-10), dim=1)
            thresh = np.percentile(entropy.flatten().detach().cpu().numpy(), self.pseudo_keep_ratio)
            thresh_mask = entropy.ge(thresh).bool()
            label_u[thresh_mask] = self.ignore_index
        return label_u

    def apply_augmentation(self, image, target, index, mix_mask=None):
        if mix_mask is None:
            mix_mask = generate_mix_masks(image, target, mode=self.pseudo_augment)
        indicator = self.indicators[index]  # (b, c)
        indicator_p = label_onehot(target, self.num_classes)
        indicator_p = indicator_p * einops.rearrange(indicator, 'b c -> b c 1 1')  # (b,c,h,w) * (b,c)
        indicator_p = indicator_p.sum(dim=1)
        mix_mask_unsqz = mix_mask.unsqueeze(1)
        image = image * mix_mask_unsqz + image.roll(-1, dims=0) * (1 - mix_mask_unsqz)
        target = target * mix_mask + target.roll(-1, dims=0) * (1 - mix_mask)
        target = target.long()
        indicator_p = indicator_p * mix_mask + indicator_p.roll(-1, dims=0) * (1 - mix_mask)
        return image, target, indicator_p

    def forward_train(self, image, target, image_u, target_u, index_u):
        #
        assert target_u is None, 'only for semi supervised learning'
        target_u = self.forward_pseudo_label(image_u)
        mix_mask = generate_mix_masks(image_u, target_u, mode=self.pseudo_augment)
        image_u, target_u, indicator_u = self.apply_augmentation(image_u, target_u, index_u, mix_mask)

        image = self.augment(image)
        image_u = self.strong_augment(image_u)
        image_all = torch.cat([image, image_u], dim=0)
        outputs = self.online_net(image_all)
        pred_all = outputs['pred']
        num_labeled, num_unlabeled = image.size(0), image_u.size(0)
        pred, pred_u = torch.split(pred_all, [num_labeled, num_unlabeled], dim=0)

        # fully supervised loss
        sup_loss = self.criterion(resize_as(pred, image), target).mean()

        # pseudo supervised loss
        pseudo_loss = self.criterion(resize_as(pred_u, image_u), target_u)
        pseudo_loss *= indicator_u
        pseudo_loss = pseudo_loss.mean()
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
