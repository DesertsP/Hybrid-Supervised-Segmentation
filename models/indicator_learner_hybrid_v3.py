import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import build_module
from models.mods.ops import point_sample, set_requires_grad, resize_as, update_model_moving_average, copy_parameters, \
    label_onehot, weighted_gather, generate_mix_masks, update_tensor_moving_average
import einops
from kornia import augmentation as augs
import numpy as np
import torch.distributed as dist
from functorch import jacrev, make_functional, vmap, grad, FunctionalModule
from functorch._src.make_functional import extract_weights
from functools import partial

if int(torch.__version__.split('.')[1]) > 12:
    from functorch import hessian
else:
    # for stability, we will not apply forward AD
    print('torch.__version__ is', torch.__version__, ', hessian computation will be slow.')
    hessian = lambda func, argnums=0: jacrev(jacrev(func, argnums), argnums)

@torch.no_grad()
def influential_sampling(logits, targets, importance_func, num_points, oversample_ratio=3, importance_sample_ratio=0.9):
    """Sample points.

    Sample points in [0, 1] x [0, 1] coordinate space based on their
    importance. The 'importance' are calculated for each point using
    'importance_func' function that takes point's logit prediction as
    input.

    Returns:
        point_coords (Tensor): A tensor of shape (batch_size, num_points,
            2) that contains the coordinates of ``num_points`` sampled
            points.
    """
    assert oversample_ratio >= 1
    assert 0 <= importance_sample_ratio <= 1
    batch_size = logits.shape[0]
    num_sampled = int(num_points * oversample_ratio)
    point_coords = torch.rand(
        batch_size, num_sampled, 2, device=logits.device)
    # point_coords_random = point_coords.clone()
    point_logits = point_sample(logits, point_coords, mode="bilinear", align_corners=False)
    point_label = point_sample(targets.unsqueeze(1).float(), point_coords, mode='nearest', align_corners=False)
    point_label = point_label.squeeze(1).long()
    # It is crucial to calculate uncertainty based on the sampled
    # prediction value for the points. Calculating uncertainties of the
    # coarse predictions first and sampling them for points leads to
    # incorrect results.  To illustrate this: assume uncertainty func(
    # logits)=-abs(logits), a sampled point between two coarse
    # predictions with -1 and 1 logits has 0 logits, and therefore 0
    # uncertainty value. However, if we calculate uncertainties for the
    # coarse predictions first, both will have -1 uncertainty,
    # and sampled point will get -1 uncertainty.
    point_importance = importance_func(point_logits, point_label)  # (B, N)
    num_important_points = int(importance_sample_ratio * num_points)
    num_random_points = num_points - num_important_points

    idx = torch.topk(
        point_importance, k=num_important_points, dim=1)[1]
    shift = num_sampled * torch.arange(
        batch_size, dtype=torch.long, device=logits.device)
    idx += shift[:, None]
    point_coords = point_coords.view(-1, 2)[idx.view(-1), :].view(
        batch_size, num_important_points, 2)
    if num_random_points > 0:
        rand_point_coords = torch.rand(
            batch_size, num_random_points, 2, device=logits.device)
        point_coords = torch.cat((point_coords, rand_point_coords), dim=1)
    return point_coords


class IndicatorLearner(nn.Module):
    def __init__(self, num_classes, num_samples,
                 indicator_initialization=1.0,
                 indicator_max=2.0,
                 indicator_lr=0.01,
                 hessian_type='influential',
                 hessian_sample_points=128,
                 ema_hessian=False,
                 ema_grad=False,
                 balance=False,
                 ema_inference=False,
                 ema_decay=0.99,
                 ema_decay_grad_hessian=0.9,
                 pseudo_augment='cutmix',
                 pseudo_keep_ratio=100,
                 network_cfg: dict = None,
                 criterion_cfg: dict = None,
                 criterion_pseudo_cfg: dict = None,
                 ignore_index=255,
                 ):
        super().__init__()
        self.pseudo_augment = pseudo_augment
        self.pseudo_keep_ratio = pseudo_keep_ratio
        self.online_net = build_module(**network_cfg)
        self.target_net = build_module(**network_cfg)
        for t_p, o_p in zip(self.target_net.parameters(), self.online_net.parameters()):
            t_p.data.copy_(o_p.data)
        set_requires_grad(self.target_net, False)

        self.criterion = nn.CrossEntropyLoss(ignore_index=ignore_index, reduction='none') if criterion_cfg is None else build_module(**criterion_cfg)
        self.criterion_pseudo = nn.CrossEntropyLoss(ignore_index=ignore_index, reduction='none') if criterion_pseudo_cfg is None else build_module(**criterion_pseudo_cfg)

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
        self.ema_inference = ema_inference
        self.ema_decay = ema_decay
        self.hessian_type = hessian_type
        self.hessian_sample_points = hessian_sample_points
        self.balance = balance
        self.ignore_index = ignore_index
        self.indicator_max = indicator_max
        self.indicator_lr = indicator_lr
        self.ema_grad = ema_grad
        self.ema_hessian = ema_hessian
        self.ema_decay_grad_hessian = ema_decay_grad_hessian

        indicators = torch.ones(num_samples, num_classes) * indicator_initialization
        self.register_buffer('indicators', indicators)
        # unbiased gradient cache
        _dim = self.online_net.classifier.out_channels * self.online_net.classifier.in_channels
        self.register_buffer('grad_buffer', torch.zeros(_dim))
        self.register_buffer('hessian_buffer', torch.zeros(_dim, _dim))

        # adam states
        m_ts = torch.zeros(num_samples, num_classes)
        v_ts = torch.zeros(num_samples, num_classes)
        ts = torch.zeros(num_samples, num_classes)
        self.register_buffer('m_ts', m_ts)
        self.register_buffer('v_ts', v_ts)
        self.register_buffer('ts', ts)

    @torch.no_grad()
    def extract_feature(self, image):
        x = self.online_net.backbone(image)
        x = self.online_net.decoder(x)
        x = self.online_net.projector(x)
        return x.detach_()

    def get_functional_classifier(self):
        classifier_functional, params = make_functional(self.online_net.classifier)
        for p in params:
            p.requires_grad_(False)
        return classifier_functional, params

    def compute_batch_grad(self, cls_func, params, feat, target):
        """
        Compute holistic gradients w.r.t classifier weights.
        Returns:
        """
        # --- define
        def compute_loss(params, inputs, targets):
            if self.balance:
                targets_oh = label_onehot(targets, num_classes=self.num_classes,
                                          ignore_index=self.ignore_index, channel_last=True)
                num_pixels_per_class = einops.reduce(targets_oh, 'b h w c -> c', reduction='sum')
                num_pixels_total = num_pixels_per_class.sum(-1, keepdim=True)
                class_weight = (num_pixels_total - num_pixels_per_class) / (1 + num_pixels_total)
            else:
                class_weight = None
            return F.cross_entropy(cls_func(params, inputs), targets,
                                   weight=class_weight, ignore_index=self.ignore_index)

        grad_func = grad(compute_loss)
        # --- compute
        batch_grad = grad_func(params, feat, target)
        batch_grad = batch_grad[0].view(-1)

        return batch_grad

    def compute_per_region_grad(self, cls_func, params, feat, target, return_pixel_grad=False):
        """
        Compute per-region gradients w.r.t classifier weights.
        For coarse/weakly labeled data point, we use the coarse labels to compute loss.
        For unlabeled data, we consider the pseudo-label loss.
        """

        # --- define
        def compute_loss(params, inputs, targets):
            # print(inputs.shape, targets.shape)
            outputs = cls_func(params, inputs)
            return F.cross_entropy(outputs, targets, reduction='mean')

        grad_per_sample_func = vmap(grad(compute_loss), in_dims=(None, 0, 0))
        # --- compute
        b, c, h, w = feat.shape
        feat = einops.rearrange(feat, 'b c h w -> (b h w) 1 c 1 1')
        _target = einops.rearrange(target.clone(), 'b h w -> (b h w) 1 1 1')
        _target[_target == self.ignore_index] = 0
        per_pixel_grad = grad_per_sample_func(params, feat, _target)
        per_pixel_grad = per_pixel_grad[0]
        per_pixel_grad = einops.rearrange(per_pixel_grad, '(b h w) o i 1 1-> b (o i) h w', b=b, h=h, w=w)
        target_oh = label_onehot(target, self.num_classes, ignore_index=self.ignore_index)  # ignore_index here
        grad_region = weighted_gather(per_pixel_grad, target_oh)  # (b, #regions, k)
        if return_pixel_grad:
            return grad_region, per_pixel_grad
        else:
            return grad_region

    def compute_hessian(self, cls_func, params, feat, target,
                        num_points=128, oversample_ratio=3, importance_sample_ratio=0.9):
        """
        NOTE: The complexity of calculating the Hessian matrix directly is terrible.
        """

        # --- define
        def compute_loss(params, inputs, targets):
            return F.cross_entropy(cls_func(params, inputs), targets)

        hessian_func = hessian(compute_loss)

        # --- compute
        logit = cls_func(params, feat)
        ce_importance_func = partial(F.cross_entropy, ignore_index=self.ignore_index, reduction='none')
        point_coords = influential_sampling(logit, target, ce_importance_func, num_points=num_points,
                                            oversample_ratio=oversample_ratio,
                                            importance_sample_ratio=importance_sample_ratio)
        feat_sampled = point_sample(feat, point_coords, mode="bilinear", align_corners=False)
        target_sampled = point_sample(target.unsqueeze(1).float(), point_coords,
                                      mode='nearest', align_corners=False).squeeze(1).long()
        target_sampled[target_sampled == self.ignore_index] = 0
        hessian_matrix = hessian_func(params, feat_sampled.unsqueeze(-1), target_sampled.unsqueeze(-1))
        hessian_matrix = hessian_matrix[0][0]
        hessian_matrix = einops.rearrange(hessian_matrix, 'i j 1 1 m n 1 1 -> (i j) (m n)')
        return hessian_matrix

    def compute_influence(self, feat, target, feat_u, target_u, sync=True):
        # --estimate influence on classifier--
        cls_func, params = self.get_functional_classifier()
        # compute regional grads on noisy data
        grad_per_region, grad_per_pixel = self.compute_per_region_grad(cls_func, params, feat_u, target_u, return_pixel_grad=True)
        b, r, k = grad_per_region.shape
        # compute batch grads on unbiased data
        grad_unbiased = self.compute_batch_grad(cls_func, params, feat, target)
        if sync:
            grad_unbiased = self.sync_tensors(grad_unbiased)
        if self.ema_grad:
            self.grad_buffer.data.copy_(
                update_tensor_moving_average(self.ema_decay_grad_hessian, self.grad_buffer, grad_unbiased).data)
        else:
            self.grad_buffer.data.copy_(grad_unbiased.data)

        grad_unbiased = self.grad_buffer.unsqueeze(0)
        if self.hessian_type == 'identity':
            influence = -grad_unbiased.matmul(einops.rearrange(grad_per_region, 'b r k -> k (b r)'))
            influence = einops.rearrange(influence, '1 (b r) ->  b r', b=b, r=r)
        elif self.hessian_type == 'influential':
            # exact hessian on influential points
            hess = self.compute_hessian(cls_func, params,
                                        torch.cat([feat, feat_u], dim=0),
                                        torch.cat([target, target_u], dim=0),
                                        num_points=self.hessian_sample_points)
            # sync hessian
            if sync:
                hess = self.sync_tensors(hess)
            if self.ema_hessian:
                self.hessian_buffer.data.copy_(
                    update_tensor_moving_average(self.ema_decay_grad_hessian, self.hessian_buffer, hess).data)
            else:
                self.hessian_buffer.data.copy_(hess.data)
            # hess_inv = self.hessian_buffer.inverse()
            hess_inv = self.hessian_buffer.inverse()
            influence = -grad_unbiased.matmul(hess_inv).matmul(einops.rearrange(grad_per_region, 'b r k -> k (b r)'))
            influence = einops.rearrange(influence, '1 (b r) ->  b r', b=b, r=r)
        elif self.hessian_type == 'covariance':
            grad_per_pixel = einops.rearrange(grad_per_pixel, 'b k h w -> k (b h w)')
            hess = grad_per_pixel @ grad_per_pixel.t()
            if sync:
                hess = self.sync_tensors(hess)
            if self.ema_hessian:
                self.hessian_buffer.data.copy_(
                    update_tensor_moving_average(self.ema_decay_grad_hessian, self.hessian_buffer, hess).data)
            else:
                self.hessian_buffer.data.copy_(hess.data)
            hess_inv = self.hessian_buffer.inverse()
            influence = -grad_unbiased.matmul(hess_inv).matmul(einops.rearrange(grad_per_region, 'b r k -> k (b r)'))
            influence = einops.rearrange(influence, '1 (b r) ->  b r', b=b, r=r)
        elif self.hessian_type == 'diagonal':
            # estimate Hessian
            raise NotImplementedError
        else:
            raise NotImplementedError
        return influence

    def compute_indicator_grad(self, image, target, image_u, target_u, index_u, sync=True):
        assert index_u is not None and target_u is not None
        if target_u is None:
            target_u = self.forward_pseudo_label(image_u)
            raise ValueError
        # extract features
        image = self.augment(image)
        image_u = self.strong_augment(image_u)
        image_all = torch.cat([image, image_u], dim=0)
        feat_all = self.extract_feature(image_all)
        num_labeled, num_unlabeled = image.size(0), image_u.size(0)
        feat, feat_u = torch.split(feat_all, [num_labeled, num_unlabeled], dim=0)
        target = resize_as(target.float().unsqueeze(1), feat, mode='nearest', align_corners=None).squeeze(1).long()
        target_u = resize_as(target_u.float().unsqueeze(1), feat_u, mode='nearest', align_corners=None).squeeze(
            1).long()
        influence = self.compute_influence(feat, target, feat_u, target_u, sync=sync)
        if self.indicators.grad is None:
            self.indicators.grad = torch.zeros_like(self.indicators)
        assert index_u.size(0) == influence.size(0)
        if sync:
            return self.sync_indicators(influence, index_u)
        else:
            self.indicators.grad[index_u] = influence
            return index_u

    @torch.no_grad()
    def adam_optimize_indicator(self, index, lr, beta1=0.9, beta2=0.999, eps=1e-8):
        indicators = self.indicators[index]
        grad_indicators = self.indicators.grad[index]
        m_t = self.m_ts[index]
        v_t = self.v_ts[index]
        t = self.ts[index]

        t += 1
        # updates the moving averages of the gradient
        m_t = beta1 * m_t + (1 - beta1) * grad_indicators
        # updates the moving averages of the squared gradient
        v_t = beta2 * v_t + (1 - beta2) * (grad_indicators * grad_indicators)
        # calculates the bias-corrected estimates
        m_cap = m_t / (1 - (beta1 ** t))
        # calculates the bias-corrected estimates
        v_cap = v_t / (1 - (beta2 ** t))

        indicators -= (lr * m_cap) / (torch.sqrt(v_cap) + eps)
        indicators.clamp_(min=0, max=self.indicator_max)

        self.indicators[index] = indicators
        self.m_ts[index] = m_t
        self.v_ts[index] = v_t
        self.ts[index] = t
        return indicators

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

    def apply_augmentation(self, image, target, index, target_aux=None, mix_mask=None):
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
        if target_aux is not None:
            target_aux = target_aux * mix_mask + target_aux.roll(-1, dims=0) * (1 - mix_mask)
            target_aux = target_aux.long()
            return image, target, target_aux, indicator_p
        else:
            return image, target, indicator_p

    def forward_train(self, image, target, image_u, target_u, index_u):
        #
        assert target_u is not None, 'only for hybrid supervised learning'
        # target_w = target_u.clone()
        target_u_pseudo = self.forward_pseudo_label(image_u)
        mix_mask = generate_mix_masks(image_u, target_u_pseudo, mode=self.pseudo_augment)
        image_u, target_u_pseudo, target_w, indicator_u = self.apply_augmentation(image_u, target_u_pseudo,
                                                                                  index_u, target_u, mix_mask)

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
        pred_u = resize_as(pred_u, image_u)
        pseudo_loss = self.criterion_pseudo(pred_u, target_u_pseudo)
        pseudo_loss *= indicator_u
        pseudo_loss = pseudo_loss.mean()
        # weakly supervised loss
        weak_loss = self.criterion_pseudo(pred_u, target_w)
        weak_loss *= indicator_u
        weak_loss = weak_loss.mean()

        # moving average
        update_model_moving_average(self.ema_decay, self.target_net, self.online_net)
        return dict(seg_loss=sup_loss, reg_loss=pseudo_loss, weak_loss=weak_loss)

    def forward_test(self, input):
        x = self.normalize(input)
        net = self.target_net if self.ema_inference else self.online_net
        pred = net(x)
        return pred

    def forward(self, *args, mode=0, **kwargs):
        if mode == 1:
            assert self.training
            return self.forward_train(*args, **kwargs)
        elif mode == 2:
            index = self.compute_indicator_grad(*args, **kwargs)
            return self.adam_optimize_indicator(index, lr=self.indicator_lr)
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
