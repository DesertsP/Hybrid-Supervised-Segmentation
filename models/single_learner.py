import torch
import torch.nn as nn
from utils import build_module
from models.mods.ops import resize_as
from kornia import augmentation as augs


class SingleLearner(nn.Module):
    def __init__(self, use_augment=False, network_cfg: dict = None):
        super().__init__()
        self.net = build_module(**network_cfg)
        self.criterion = nn.CrossEntropyLoss(ignore_index=255)

        self.normalize = augs.Normalize(mean=torch.tensor((0.485, 0.456, 0.406)),
                                        std=torch.tensor((0.229, 0.224, 0.225)))
        DEFAULT_AUG = nn.Sequential(
            augs.ColorJitter(0.5, 0.5, 0.5, 0.25, p=0.8),
            augs.RandomGrayscale(p=0.2),
            augs.RandomGaussianBlur((3, 3), (0.1, 2.0), p=0.5),
            augs.RandomSolarize(p=0.1),
            self.normalize
        )
        self.augment = DEFAULT_AUG if use_augment else self.normalize

    def forward_train(self, input, target):
        pred = self.net(self.augment(input))['pred']
        reg_loss = 0.0
        seg_loss = self.criterion(pred, target)
        return dict(seg_loss=seg_loss, reg_loss=reg_loss)

    def forward_test(self, input):
        x = self.normalize(input)
        pred = self.net(x)
        return pred

    def forward(self, input, target=None):
        if not self.training:
            return self.forward_test(input)
        return self.forward_train(input, target)

    def parameter_groups(self):
        groups = ([], [])
        backbone = [self.net.backbone]
        newly_added = [self.net.decoder,
                       self.net.classifier, ]
        for module in backbone:
            for p in module.parameters():
                groups[0].append(p)
        for module in newly_added:
            for p in module.parameters():
                groups[1].append(p)
        assert len(list(self.parameters())) == sum([len(g) for g in groups])
        return groups

