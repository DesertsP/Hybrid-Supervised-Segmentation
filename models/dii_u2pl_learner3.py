import copy
import torch
import torch.nn as nn
from utils import build_module
from models.mods.ops import set_requires_grad, resize_as, update_model_moving_average
from kornia import augmentation as augs
from datasets.augmentation import generate_unsup_data
import numpy as np
import torch.nn.functional as F
import einops
from models.deeplab_proj import DeepLab
from models.losses.u2pl import compute_contra_memobank_loss
from backpack import extend, backpack
from backpack.extensions import BatchGrad
from functools import wraps


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
        thresh = np.percentile(
            entropy[target != 255].detach().cpu().numpy().flatten(), percent
        )
        thresh_mask = entropy.ge(thresh).bool() * (target != 255).bool()

        target[thresh_mask] = 255
        weight = batch_size * h * w / torch.sum(target != 255)

    loss = weight * F.cross_entropy(predict, target, ignore_index=255)

    return loss


def weighted_segmentation_loss(predict, target, weight, percent, entropy, gamma=1):
    """
    weight: (b, h, w)
    """
    batch_size, num_class, h, w = predict.shape

    with torch.no_grad():
        # drop pixels with high entropy
        thresh = np.percentile(
            entropy[target != 255].detach().cpu().numpy().flatten(), percent
        )
        thresh_mask = entropy.ge(thresh).bool() * (target != 255).bool()

        target[thresh_mask] = 255
        # weight = batch_size * h * w / torch.sum(target != 255)
        weight = torch.clamp(weight, min=0)
        weight[thresh_mask] = 0.
        weight /= torch.sum(weight) + 1.
        weight = weight ** gamma
    loss = weight * F.cross_entropy(predict, target, ignore_index=255, reduction='none')
    loss = loss.sum()
    return loss


def balanced_segmentation_loss(predict, target, num_classes):
    """
    weight: (b, h, w)
    """
    with torch.no_grad():
        target_one_hot = label_onehot(target, num_classes)
        num_pixels_per_class = einops.reduce(target_one_hot, 'b c h w -> c', reduction='sum')
        num_pixels_total = num_pixels_per_class.sum(-1, keepdim=True)
        class_weight = (num_pixels_total - num_pixels_per_class) / (1 + num_pixels_total)
    loss = F.cross_entropy(predict, target, weight=class_weight, ignore_index=255, reduction='mean')
    return loss


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


class DiiU2plLearner(nn.Module):
    def __init__(self, num_classes, ema_decay=0.99, use_augment=False, hessian_type='identity',
                 network_cfg: dict = None):
        super().__init__()
        self.online_net = build_module(**network_cfg)
        assert isinstance(self.online_net, DeepLab)
        self.target_net = build_module(**network_cfg)
        for t_p, o_p in zip(self.target_net.parameters(), self.online_net.parameters()):
            t_p.data.copy_(o_p.data)
        set_requires_grad(self.target_net, False)

        self.proxy_classifier = None
        self.hessian_type = hessian_type
        self.proxy_criterion = extend(nn.CrossEntropyLoss(ignore_index=255, reduction='none'))
        self.criterion = nn.CrossEntropyLoss(ignore_index=255)

        self.normalize = augs.Normalize(mean=torch.tensor((0.485, 0.456, 0.406)),
                                        std=torch.tensor((0.229, 0.224, 0.225)))
        DEFAULT_AUG = nn.Sequential(
            augs.ColorJitter(0.5, 0.5, 0.5, 0.1, p=0.5),
            self.normalize
        )
        self.augment = DEFAULT_AUG if use_augment else self.normalize
        self.cfg = {'low_rank': 3,
                    'high_rank': 20,
                    'current_class_threshold': 0.3,
                    'current_class_negative_threshold': 1,
                    'unsupervised_entropy_ignore': 80,
                    'low_entropy_threshold': 20,
                    'num_negatives': 50,
                    'num_queries': 256,
                    'temperature': 0.5}
        prototypes = torch.zeros(num_classes, self.cfg['num_queries'], 1, 256)
        self.register_buffer('prototypes', prototypes)
        self.memory_bank = [[torch.zeros(0, 256)] for _ in range(num_classes)]
        self.queue_size = [30000] * num_classes
        self.queue_size[0] = 50000
        self.queue_ptrlis = [torch.zeros(1, dtype=torch.long) for _ in range(num_classes)]
        self.num_classes = num_classes
        self.ema_decay = ema_decay

        global_grad = torch.zeros(1, self.online_net.classifier.out_channels * self.online_net.classifier.in_channels)
        self.register_buffer('global_grad', global_grad)

    @torch.no_grad()
    def forward_pseudo_label(self, image_u):
        # generate pseudo labels first
        self.target_net.eval()
        pred_u_teacher = self.target_net(image_u)["pred"]
        pred_u_teacher = resize_as(pred_u_teacher, image_u)
        pred_u_teacher = pred_u_teacher.softmax(dim=1)
        logits_u, label_u = torch.max(pred_u_teacher, dim=1)
        # apply strong data augmentation: cutout, cutmix, or classmix
        if np.random.uniform(0, 1) < 0.5:
            image_u, label_u, logits_u = generate_unsup_data(image_u, label_u.clone(), logits_u.clone(),
                                                             mode='cutmix')
        return image_u, label_u

    @singleton('proxy_classifier')
    def _get_proxy_classifier(self):
        proxy_classifier = copy.deepcopy(self.online_net.classification_loss)
        proxy_classifier = extend(proxy_classifier)
        return proxy_classifier

    def grad_per_instance(self, representations, targets):
        """
        Compute per-sample gradients w.r.t the proxy classifier.
        For coarse/weakly labeled data point, we can use the coarse labels to compute loss.
        For unlabeled data, we can consider the regularization/pseudo-label loss.
        Args:
            representations:
            targets:
        Returns:
        """
        proxy_classifier = self._get_proxy_classifier()
        # for t_p, o_p in zip(proxy_classifier.parameters(), self.online_net.classifier.parameters()):
        #     t_p.data.copy_(o_p.data)
        b, c, h, w = representations.shape
        representations = einops.rearrange(representations, 'b c h w -> (b h w) c 1 1')
        pred = proxy_classifier(representations)
        pred = einops.rearrange(pred, '(b h w) c 1 1 -> b c h w', b=b, h=h, w=w)

        loss = self.proxy_criterion(resize_as(pred, targets), targets).sum()
        with backpack(BatchGrad()):
            loss.backward()
        grad = proxy_classifier.weight.grad_batch.detach().clone()
        # grad = einops.rearrange(grad, '(b h w) o i 1 1-> b (o i) h w', b=b, h=h, w=w)
        grad = einops.rearrange(grad, 'n o i 1 1-> n (o i)')

        # assert torch.allclose(proxy_classifier.weight.grad, grad.sum(dim=0))
        proxy_classifier.zero_grad()
        return grad

    def grad_batch(self, representations, targets):
        proxy_classifier = self._get_proxy_classifier()
        # for t_p, o_p in zip(proxy_classifier.parameters(), self.online_net.classifier.parameters()):
        #     t_p.data.copy_(o_p.data)
        pred = proxy_classifier(representations)
        # loss = balanced_segmentation_loss(resize_as(pred, targets), targets, self.num_classes)
        loss = self.criterion(resize_as(pred, targets), targets)
        loss.backward()
        grad = proxy_classifier.weight.grad.detach().clone()
        proxy_classifier.zero_grad()
        return grad.reshape(1, -1)  # (1, #i*#o)

    def hessian(self):
        proxy_classifier = self._get_proxy_classifier()
        # for t_p, o_p in zip(proxy_classifier.parameters(), self.online_net.classifier.parameters()):
        #     t_p.data.copy_(o_p.data)
        pred = proxy_classifier(representations)
        # loss = balanced_segmentation_loss(resize_as(pred, targets), targets, self.num_classes)
        loss = self.criterion(resize_as(pred, targets), targets)
        loss.backward()
        grad = proxy_classifier.weight.grad.detach().clone()
        proxy_classifier.zero_grad()

    def influence(self, representations_l, targets_l, representations_u, targets_u):
        proxy_classifier = self._get_proxy_classifier()
        for t_p, o_p in zip(proxy_classifier.parameters(), self.online_net.classification_loss.parameters()):
            t_p.data.copy_(o_p.data)
        grad_l = self.grad_batch(representations_l, targets_l)  # (1, c), c=#i*#o
        grad_u = self.grad_per_instance(representations_u, targets_u)  # (bhw, c)
        with torch.no_grad():
            self.global_grad = (1 - self.ema_decay) * grad_l + self.ema_decay * self.global_grad
            if self.hessian_type == 'identity':
                influence = self.global_grad.matmul(grad_u.permute(1, 0))  # - dl / d \gamma
            else:
                # compute / estimate inverse Hessian
                raise NotImplementedError

            # normalize
            # influence = torch.clamp(influence, min=0)
            # influence /= torch.sum(influence, dim=0, keepdim=True) + 1.
            # reshape
            b, _, h, w = representations_u.shape
            influence = einops.rearrange(influence, '1 (b h w) -> b 1 h w', b=b, h=h, w=w)
            influence = resize_as(influence, targets_u, mode='bilinear', align_corners=True).squeeze(1)
            grad_u_norm = torch.norm(grad_u, p=1, dim=1, keepdim=True)
            grad_u_norm = einops.rearrange(grad_u_norm, '(b h w) 1 -> b 1 h w', b=b, h=h, w=w)
            grad_u_norm = resize_as(grad_u_norm, targets_u, mode='bilinear', align_corners=True).squeeze(1)
        return influence, grad_u_norm

    def forward_train(self, image, target, image_u):
        image = self.augment(image)
        image_u = self.augment(image_u)
        image_u, target_u = self.forward_pseudo_label(image_u)
        image_all = torch.cat([image, image_u], dim=0)
        outputs = self.online_net(image_all)
        pred_all, proj_all = outputs['pred'], outputs['repr_con']
        num_labeled, num_unlabeled = image.size(0), image_u.size(0)
        pred, pred_u = torch.split(pred_all, [num_labeled, num_unlabeled], dim=0)
        repr_cls, repr_cls_u = torch.split(outputs['repr_cls'], [num_labeled, num_unlabeled], dim=0)

        # supervised loss
        sup_loss = self.criterion(resize_as(pred, image), target)
        # sup_loss = balanced_segmentation_loss(resize_as(pred, image), target, self.num_classes)
        # teacher forward
        self.target_net.train()
        with torch.no_grad():
            outputs_t = self.target_net(image_all)
            pred_all_teacher, proj_all_teacher = outputs_t['pred'], outputs_t['repr_con']
            prob_all_teacher = pred_all_teacher.softmax(dim=1)
            prob_l_teacher, prob_u_teacher = torch.split(pred_all, [num_labeled, num_unlabeled], dim=0)
            pred_u_teacher = pred_all_teacher[num_labeled:]

        pixel_influence, entropy = self.influence(repr_cls.detach().clone(), target,
                                                  repr_cls_u.detach().clone(), target_u)
        # unsupervised loss
        # pseudo_loss = pseudo_segmentation_loss(resize_as(pred_u, target_u), target_u.clone(), 80, entropy)
        pseudo_loss = weighted_segmentation_loss(resize_as(pred_u, target_u), target_u.clone(),
                                                 weight=pixel_influence, percent=80, entropy=entropy, gamma=1)

        # contrastive
        with torch.no_grad():
            # prob = torch.softmax(resize_as(pred_u_teacher, target_u), dim=1)
            # entropy = -torch.sum(prob * torch.log(prob + 1e-10), dim=1)
            low_thresh = np.percentile(entropy[target_u != 255].cpu().numpy().flatten(),
                                       self.cfg['low_entropy_threshold'])
            low_entropy_mask = entropy.le(low_thresh).float() * (target_u != 255).bool()
            high_thresh = np.percentile(entropy[target_u != 255].cpu().numpy().flatten(),
                                        100 - self.cfg['low_entropy_threshold'])
            high_entropy_mask = entropy.ge(high_thresh).float() * (target_u != 255).bool()

            # low uncertainty / high confidence mask. all labeled samples are high confidence
            low_entropy_mask_all = torch.cat([(target.unsqueeze(1) != 255).float(), low_entropy_mask.unsqueeze(1)])
            # down sample
            low_entropy_mask_all = resize_as(low_entropy_mask_all, pred_all, mode='nearest', align_corners=None)
            # samples with higher uncertainty will be selected as negatives
            high_entropy_mask_all = torch.cat([(target.unsqueeze(1) != 255).float(), high_entropy_mask.unsqueeze(1)])
            high_entropy_mask_all = resize_as(high_entropy_mask_all, pred_all, mode='nearest', align_corners=None)
            # down sample targets, & one-hot label
            target_one_hot = label_onehot(target, num_classes=self.num_classes)
            target_u_one_hot = label_onehot(target_u, num_classes=self.num_classes)

        new_keys, contra_loss = compute_contra_memobank_loss(
            proj_all,
            resize_as(target_one_hot.float(), pred_all, mode='nearest', align_corners=None).long(),
            resize_as(target_u_one_hot.float(), pred_all, mode='nearest', align_corners=None).long(),
            prob_l_teacher.detach(),
            prob_u_teacher.detach(),
            low_entropy_mask_all,
            high_entropy_mask_all,
            self.cfg,
            self.memory_bank,
            self.queue_ptrlis,
            self.queue_size,
            proj_all_teacher.detach(),
        )

        return dict(seg_loss=sup_loss + pseudo_loss, reg_loss=contra_loss)

    def forward_test(self, input):
        x = self.normalize(input)
        pred = self.online_net(x)
        return pred

    def forward(self, input, target=None, input_u=None):
        if not self.training:
            return self.forward_test(input)
        loss = self.forward_train(input, target, input_u)
        update_model_moving_average(self.ema_decay, self.target_net, self.online_net)
        return loss

    def parameter_groups(self):
        return self.online_net.parameter_groups()
