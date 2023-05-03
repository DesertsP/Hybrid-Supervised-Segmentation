import torch
import torch.nn as nn
from utils import build_module
from models.mods.ops import resize_as
from kornia import augmentation as augs


class CrossPseudoLearner(nn.Module):
    def __init__(self, use_augment=False, network_cfg: dict = None):
        super().__init__()
        self.branch_one = build_module(**network_cfg)
        self.branch_two = build_module(**network_cfg)
        self.criterion = nn.CrossEntropyLoss(reduction='mean', ignore_index=255)

        self.normalize = augs.Normalize(mean=torch.tensor((0.485, 0.456, 0.406)),
                                        std=torch.tensor((0.229, 0.224, 0.225)))
        DEFAULT_AUG = nn.Sequential(
            augs.ColorJitter(0.5, 0.5, 0.5, 0.1, p=0.5),
            # augs.RandomGaussianBlur((3, 3), (1.5, 1.5), p=0.5),
            # augs.RandomSolarize(p=0.1),
            self.normalize
        )
        self.augment = DEFAULT_AUG if use_augment else self.normalize

    def forward_train(self, input, target=None):
        x_one = self.augment(input)
        x_two = self.augment(input)
        pred_one = self.branch_one(x_one)['pred']
        pred_two = self.branch_two(x_two)['pred']
        # cps
        pseudo_one = torch.argmax(pred_one, dim=1).long()
        pseudo_two = torch.argmax(pred_two, dim=1).long()
        reg_loss = self.criterion(pred_one, pseudo_two) + self.criterion(pred_two, pseudo_one)

        pred_one = resize_as(pred_one, x_one)
        pred_two = resize_as(pred_two, x_two)
        # seg loss
        if target is not None:
            seg_loss = self.criterion(pred_one, target) + self.criterion(pred_two, target)
        else:
            seg_loss = 0.0
        return dict(seg_loss=seg_loss, reg_loss=reg_loss)

    def forward_test(self, input):
        x = self.normalize(input)
        pred = self.branch_one(x)
        return pred

    def forward(self, input, target=None, input_u=None):
        if not self.training:
            return self.forward_test(input)
        loss = self.forward_train(input, target)
        if input_u is not None:
            loss['reg_loss'] += self.forward_train(input_u)['reg_loss']
        return loss

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
    net.train()
    net.forward(torch.randn(2, 3, 128, 128), torch.randn(2, 3, 128, 128), torch.zeros(2, 128, 128, dtype=torch.long))
