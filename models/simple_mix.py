from kornia import augmentation as augs
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import build_module, get_module
from models.mods.ops import generate_mix_masks
from models.mods.ops import resize_as
from models.deeplab2 import DeepLab


class SimpleMixLearner(DeepLab):
    def __init__(self, ignore_index=255, use_color_augment=False, augment_type='cutmix', supervised_only=False, **kwargs):
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
        self.strong_augment = STRONG_AUG if use_color_augment else DEFAULT_AUG
        self.normalize = NORMALIZE

        self.criterion = nn.CrossEntropyLoss(ignore_index=ignore_index)
        self.augment_type = augment_type
        self.supervised_only = supervised_only

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
            return super().forward(x)
        else:
            image = self.augment(image)

            if self.supervised_only:
                pred = super().forward(image)['pred']
                loss = self.criterion(resize_as(pred, target), target)
                return dict(seg_loss=loss)

            image_u = self.strong_augment(image_u)
            if self.augment_type:
                image_u, target_u = self.apply_augmentation(image_u, target_u)
            pred = super().forward(torch.cat([image, image_u]))['pred']
            target_all = torch.cat([target, target_u])
            loss = self.criterion(resize_as(pred, target_all), target_all)
            return dict(seg_loss=loss)

    def parameter_groups(self):
        groups = ([], [])
        backbone = [self.backbone]
        newly_added = [self.decoder,
                       self.projector,
                       self.classifier
                       ]
        for module in backbone:
            for p in module.parameters():
                groups[0].append(p)
        for module in newly_added:
            for p in module.parameters():
                groups[1].append(p)
        if self.aux_output:
            newly_added.append(self.aux_classifier)
        assert len(list(self.parameters())) == sum([len(g) for g in groups])
        return groups
