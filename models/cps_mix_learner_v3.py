import torch
import torch.nn as nn
from utils import build_module
from models.mods.ops import resize_as
from kornia import augmentation as augs
import random


class CrossPseudoLearner(nn.Module):
    def __init__(self, use_augment=False, mix_prob=1.0, network_cfg: dict = None):
        super().__init__()
        self.branch_one = build_module(**network_cfg)
        self.branch_two = build_module(**network_cfg)
        self.criterion = nn.CrossEntropyLoss(reduction='mean', ignore_index=255)

        self.normalize = augs.Normalize(mean=torch.tensor((0.485, 0.456, 0.406)),
                                        std=torch.tensor((0.229, 0.224, 0.225)))
        DEFAULT_AUG = nn.Sequential(
            augs.ColorJitter(0.5, 0.5, 0.5, 0.1, p=0.8),
            # augs.RandomGaussianBlur((3, 3), (1.5, 1.5), p=0.5),
            # augs.RandomSolarize(p=0.1),
            self.normalize
        )
        self.augment = DEFAULT_AUG if use_augment else self.normalize
        self.mix_prob = mix_prob

    @torch.no_grad()
    def forward_mix(self, input, mask=None):
        """
        input: (b,h,w) or (b,c,h,w)
        mask: (b,h,w)
        """
        if len(mask.shape) != len(input.shape):
            mask = mask.unsqueeze(1)
        mask = mask.type_as(input)
        input = input * mask + input.roll(shifts=input.size(0)//2, dims=0) * (1 - mask)
        return input

    @torch.no_grad()
    def forward_pseudo(self, input, branch=1):
        # generate pseudo labels
        input = self.normalize(input)
        assert branch in (1, 2)
        if branch == 1:
            pred = self.branch_one(input)['pred']
        else:
            pred = self.branch_two(input)['pred']
        pred = resize_as(pred, input)
        pseudo = torch.argmax(pred, dim=1).long()
        return pseudo

    def forward_loss(self, input, target, mask, branch=1):
        if random.random() < self.mix_prob:
            input = self.forward_mix(input, mask)
            target = self.forward_mix(target, mask)
        input = self.augment(input)
        assert branch in (1, 2)
        if branch == 1:
            pred = self.branch_one(input)['pred']
        else:
            pred = self.branch_two(input)['pred']
        pred = resize_as(pred, input)
        loss = self.criterion(pred, target)
        return loss

    def forward_train(self, input, target=None, mask=None, input_u=None, mask_u=None):
        pseudo_one = self.forward_pseudo(input_u, branch=1)
        pseudo_two = self.forward_pseudo(input_u, branch=2)
        input_cat = torch.cat([input, input_u], dim=0)
        mask_cat = torch.cat([mask, mask_u], dim=0)
        target_cat_one = torch.cat([target, pseudo_two], dim=0)
        target_cat_two = torch.cat([target, pseudo_one], dim=0)
        seg_loss_one = self.forward_loss(input_cat, target_cat_one, mask_cat, branch=1)
        seg_loss_two = self.forward_loss(input_cat, target_cat_two, mask_cat, branch=2)
        return dict(seg_loss=seg_loss_one+seg_loss_two)

    def forward_test(self, input):
        x = self.normalize(input)
        pred = self.branch_one(x)
        return pred

    def forward(self, input, target=None, mask=None, input_u=None, mask_u=None):
        if not self.training:
            return self.forward_test(input)
        return self.forward_train(input, target, mask, input_u, mask_u)

    def parameter_groups(self):
        groups = ([], [])
        backbone = [self.branch_one.backbone, self.branch_two.backbone]
        newly_added = [self.branch_one.decoder, self.branch_two.decoder,
                       self.branch_one.classification_loss, self.branch_two.classification_loss]
        for module in backbone:
            for p in module.parameters():
                groups[0].append(p)
        for module in newly_added:
            for p in module.parameters():
                groups[1].append(p)
        assert len(list(self.parameters())) == sum([len(g) for g in groups])
        return groups


if __name__ == '__main__':
    net = CrossPseudoLearner(network_cfg={'name': 'models.deeplab.DeepLab', 'num_classes': 21,
                                          'backbone_cfg': {'name': 'models.backbones.resnet.resnet101',
                                                           'pretrained': False,
                                                           'replace_stride_with_dilation': [False, True, True],
                                                           'zero_init_residual': True},
                                          'decoder_cfg': {'name': 'models.mods.aspp.ASPPDecoder'},
                                          'norm_cfg': {'name': 'torch.nn.BatchNorm2d'}})
    net.eval()
    net.forward(torch.randn(2, 3, 128, 128))
