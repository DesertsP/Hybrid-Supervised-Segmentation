import torch
import torch.nn as nn
import torch.nn.functional as F

from models.deeplab import SingleNetwork
from models.mods.augmentation import RandomApply
from kornia import augmentation as augs

import utils.autograd_hacks as autograd_hacks
import utils.hessian as hessian


class Net(nn.Module):
    """
    Example:
    >>> net = Net()
    >>> x1 = torch.randn(4, 3, 448, 448)
    >>> x2 = torch.randn_like(x1)
    >>> ind1 = torch.ones(4, dtype=torch.long)
    >>> y1 = torch.zeros(4, 448, 448, dtype=torch.long)
    >>> net(x1, y1, ind1, x2, y1, ind1, y1[:, None], mode=1)
    """
    def __init__(self, use_augment=False, loss_cls_weight=None, loss_coefficient=(1.0, 1.0),
                 num_samples=int(1e5), sample_weight_init=1.0, hessian_type='hessian',
                 mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5), **kwargs):
        super().__init__()
        assert len(loss_coefficient) == 2, 'there are two kinds of losses: seg loss and pseudo (regularization) loss.'
        assert 0 <= sample_weight_init <= 1.0
        assert hessian_type in ['identity', 'hessian']
        # Dual networks with different init.
        self.branch_one = SingleNetwork(**kwargs) 
        self.branch_two = SingleNetwork(**kwargs)

        DEFAULT_AUG = nn.Sequential(
            RandomApply(augs.ColorJitter(0.3, 0.3, 0.3, 0.1), p=0.8),
            augs.RandomGaussianBlur((3, 3), (1.5, 1.5), p=0.3),
            # augs.RandomSolarize(p=0.1),
            augs.Normalize(mean=torch.tensor(mean), std=torch.tensor(std))
        )

        self.augment_val = augs.Normalize(mean=torch.tensor(mean), std=torch.tensor(std))
        self.augment = DEFAULT_AUG if use_augment else self.augment_val

        self.criterion = nn.CrossEntropyLoss(weight=torch.Tensor(loss_cls_weight) if loss_cls_weight else None,
                                             ignore_index=255, reduction='none')
        # self.loss_coefficient = loss_coefficient    # for gradient update

        # --
        # for per-sample weight learning, hack the backprop. to obtain gradients w.r.t each sample
        self.register_buffer('per_sample_weight', torch.ones(num_samples)*sample_weight_init)
        self.hessian_type = hessian_type
        autograd_hacks.add_hooks(self.branch_one.classification_loss)    # capture per-sample gradients, only considering weights of last layer
        autograd_hacks.disable_hooks()
        self.grad_val = None
        # --

    def parameter_groups(self):
        groups = list(self.branch_one.parameter_groups())
        for i, g in enumerate(self.branch_two.parameter_groups()):
            groups[i] += g
        return groups

    def forward_train(self, input_i, target_i, indices_i, input_j, target_j, indices_j, mask):
        x_i = self.augment(input_i)
        x_j = self.augment(input_j)
        # mix inputs
        mask = mask.type_as(x_i)
        x_mix = x_i * (1 - mask) + x_j * mask
        x_i_j = torch.cat([x_i, x_j], dim=0)
        # assert x_i.shape[0] == x_j.shape[0]
        # Estimate the pseudo-label with branch#1 & supervise branch#2
        _, pred_one = self.branch_one(x_i_j)
        pred_one_i, pred_one_j = pred_one.chunk(2, dim=0)
        pred_one = pred_one_i * (1 - mask) + pred_one_j * mask
        pseudo_one = torch.argmax(pred_one, dim=1).long()

        # Estimate the pseudo-label with branch#2 & supervise branch#1
        _, pred_two = self.branch_two(x_i_j)
        pred_two_i, pred_two_j = pred_two.chunk(2, dim=0)
        pred_two = pred_two_i * (1 - mask) + pred_two_j * mask
        pseudo_two = torch.argmax(pred_two, dim=1).long()

        # weights
        w_i = self.per_sample_weight[indices_i]
        w_j = self.per_sample_weight[indices_j]
        w_i = w_i[:, None, None]
        w_j = w_j[:, None, None]
        w_mix = (w_i + w_j) / 2.0
        # pseudo loss for mixed data
        _, pred_one = self.branch_one(x_mix)
        _, pred_two = self.branch_two(x_mix)
        cps_loss = self.criterion(pred_one, pseudo_two) + self.criterion(pred_two, pseudo_one)
        cps_loss *= w_mix
        cps_loss = cps_loss.mean()
        # seg loss
        if target_i is not None and target_j is not None:
            seg_loss_i = self.criterion(pred_one_i, target_i) + self.criterion(pred_two_i, target_i)
            seg_loss_j = self.criterion(pred_one_j, target_j) + self.criterion(pred_two_j, target_j)
            seg_loss_i *= w_i
            seg_loss_j *= w_j
            seg_loss = seg_loss_i.mean() + seg_loss_j.mean()
        else:
            seg_loss = 0
        # 对于一个样本loss来自: segloss(pred, label)
        # mix的样本算出来的persample gradient是对于两个样本的，无法重新分配给两个样本。
        return seg_loss, cps_loss
    
    def forward_test(self, input):
        x = self.augment_val(input)
        pred = self.branch_one(x)
        return pred

    def forward(self, *args, mode=0, **kwargs):
        """
        dispatch: 
        1. training -> forward_train
        2. compute overall gradients -> compute_grad_val
        3. compute per-sample gradients -> update_weight_grad
        0. test mode (for validation and test) -> forward_test
        """
        if mode == 1 and self.training:
            return self.forward_train(*args, **kwargs)
        elif mode == 2:
            # update unbiased gradients
            return self.compute_grad_val(*args, **kwargs)
        elif mode == 3:
            # update per-sample gradients & influence
            return self.update_weight_grad(*args, **kwargs)
        else:
            return self.forward_test(*args, **kwargs)

    def compute_grad_val(self, inputs, targets, update_cache=True):
        grads = self.grad_classifier(inputs, targets)
        if update_cache:
            del self.grad_val
            self.grad_val = grads
        return grads

    def update_weight_grad(self, inputs, targets, indices, inputs_val=None, targets_val=None):
        if inputs_val and targets_val:
            self.compute_grad_val(inputs_val, targets_val)
        influence = self.influence(inputs, targets, indices)

        if self.per_sample_weight.grad is None:
            self.per_sample_weight.grad = torch.zeros_like(self.per_sample_weight)
        self.per_sample_weight.grad[indices] = influence  # store grads for current samples only
        return influence    # influence of data in current batch

    def grad_classifier(self, inputs, targets):
        # we only use classifier layer of branch_one to compute grads
        x = self.augment_val(inputs)
        pred = self.forward_classifier(x)
        loss = self.criterion(pred, targets).mean()
        loss.backward()
        grad = self.branch_one.classification_loss.weight.grad.detach().clone()
        self.branch_one.classification_loss.zero_grad()
        return grad.reshape(1, -1)
    
    def grad_per_sample_classifier(self, inputs, targets):
        """
        Compute per-sample gradients.
        For coarse/weakly labeled data point, we can use the coarse labels to compute loss.
        For unlabeled data, we can consider the regularization/pseudo-label loss.
        Args:
            inputs: coarse inputs
            targets: coarse targets
        Returns:
        """
        x = self.augment_val(inputs)
        autograd_hacks.enable_hooks()
        autograd_hacks.clear_backprops(self.branch_one.classification_loss)
        pred_one = self.forward_classifier(x)
        loss = self.criterion(pred_one, targets).mean()
        loss.backward(retain_graph=True)
        # 
        autograd_hacks.compute_grad1(self.branch_one.classification_loss)
        autograd_hacks.disable_hooks()
        per_sample_grad = self.branch_one.classification_loss.weight.grad1.detach().clone()
        # assert torch.allclose(self.branch_one.classifier.weight.grad, per_sample_grad.mean(dim=0))
        self.branch_one.classification_loss.zero_grad()
        return per_sample_grad.reshape(inputs.shape[0], -1)     # flatten
        
    def forward_classifier(self, inputs):
        # helper function which turn off gradients or cache on layers except classifier
        # use branch_one only
        # self.eval()     # don't forget to turn off running average of bn
        # forward
        with torch.no_grad():
            blocks = self.branch_one.backbone(inputs)
            v3plus_feature = self.branch_one.head(blocks).detach()  # (b, c, h, w)
        pred = self.branch_one.classification_loss(v3plus_feature)
        b, c, h, w = inputs.shape
        pred = F.interpolate(pred, size=(h, w), mode='bilinear', align_corners=True)
        return pred

    def influence(self, inputs, targets, indices, grad_val=None):
        if grad_val is None:
            # use cached gradients
            grad_val = self.grad_val
            assert grad_val is not None, 'must compute unbiased gradients first.'
        grad_per_sample = self.grad_per_sample_classifier(inputs, targets)
        if self.hessian_type == 'identity':
            influence = -grad_val.matmul(grad_per_sample.permute(1, 0))
        else:
            # compute / estimate inverse Hessian
            h_ = self.hessian_classifier(inputs, targets, indices)
            h_inv = h_.inverse()
            # influence
            influence = -grad_val.matmul(h_inv).matmul(grad_per_sample.permute(1, 0))
        return influence.reshape(-1)

    def hessian_classifier(self, inputs, targets, indices):
        """
        TODO: The complexity of calculating the Hessian matrix directly is terrible.
        Args:
            inputs:
            targets:
            indices:
        Returns:
        """
        raise NotImplementedError


if __name__ == "__main__":
    import doctest
    doctest.testmod()