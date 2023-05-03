from kornia import augmentation as augs
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import build_module, get_module
from models.mods.ops import generate_mix_masks
from models.mods.ops import resize_as


class StrongWeakNetwork(nn.Module):
    def __init__(self, num_classes, backbone_cfg: dict, norm_cfg: dict):
        super(StrongWeakNetwork, self).__init__()
        norm_layer = get_module(**norm_cfg)
        backbone_cfg.update({'norm_layer': norm_layer})
        self.backbone = build_module(**backbone_cfg)

        self.neck = nn.Sequential(nn.Conv2d(2048, 256, kernel_size=3, stride=1, padding=1, bias=False),
                                  norm_layer(256),
                                  nn.ReLU(),
                                  nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False),
                                  norm_layer(256),
                                  nn.ReLU(),
                                  )

        self.head_one = nn.Sequential(nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False),
                                      norm_layer(256),
                                      nn.ReLU(),
                                      nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False),
                                      norm_layer(256),
                                      nn.ReLU(),
                                      nn.Conv2d(256, num_classes, kernel_size=1, bias=True),
                                      )
        self.head_two = nn.Sequential(nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False),
                                      norm_layer(256),
                                      nn.ReLU(),
                                      nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False),
                                      norm_layer(256),
                                      nn.ReLU(),
                                      nn.Conv2d(256, num_classes, kernel_size=1, bias=True),
                                      )
        self.business_layer = [self.neck, self.head_one, self.head_two]

        for m in self.business_layer:
            self._init_weight(m)

    def forward(self, x):
        blocks = self.backbone(x)
        feats = self.neck(blocks[-1])
        pred1 = self.head_one(feats)
        pred2 = self.head_two(feats)
        return pred1, pred2

    @staticmethod
    def _init_weight(module):
        for m in module.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.SyncBatchNorm):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def parameter_groups(self):
        groups = ([], [])
        backbone = [self.backbone]
        newly_added = self.business_layer
        for module in backbone:
            for p in module.parameters():
                groups[0].append(p)
        for module in newly_added:
            for p in module.parameters():
                groups[1].append(p)
        assert len(list(self.parameters())) == sum([len(g) for g in groups])
        return groups


class StrongWeakLearner(StrongWeakNetwork):
    def __init__(self, ignore_index=255, augment_type='cutmix', **kwargs):
        super().__init__(**kwargs)
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

        self.criterion = nn.CrossEntropyLoss(ignore_index=ignore_index)
        self.augment_type = augment_type

    def apply_augmentation(self, image, target, mix_mask=None):
        if mix_mask is None:
            mix_mask = generate_mix_masks(image, target, mode=self.augment_type)
        mix_mask_unsqz = mix_mask.unsqueeze(1)
        image = image * mix_mask_unsqz + image.roll(-1, dims=0) * (1 - mix_mask_unsqz)
        target = target * mix_mask + target.roll(-1, dims=0) * (1 - mix_mask)
        target = target.long()
        return image, target

    def forward(self, image, target=None, image_u=None, target_u=None, *args, **kwargs):
        if not self.training:
            x = self.normalize(image)
            pred = super().forward(x)[0]
            return dict(pred=pred)
        else:
            image = self.augment(image)
            image_u = self.strong_augment(image_u)
            if self.augment_type:
                image_u, target_u = self.apply_augmentation(image_u, target_u)
            pred_one, pred_two = super().forward(torch.cat([image, image_u]))
            pred_one1, pred_one2 = torch.split(pred_one, [image.size(0), image_u.size(0)], dim=0)
            pred_two1, pred_two2 = torch.split(pred_two, [image.size(0), image_u.size(0)], dim=0)
            loss = self.criterion(resize_as(pred_one1, target), target) + \
                   self.criterion(resize_as(pred_two2, target_u), target_u)
            return dict(seg_loss=loss)
