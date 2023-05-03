import torch
import torch.nn as nn
from utils import build_module
from models.mods.ops import set_requires_grad, resize_as, update_model_moving_average
from kornia import augmentation as augs
from datasets.augmentation import generate_unsup_data
import numpy as np
from torch.nn import functional as F


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
    def __init__(self, num_classes, ema_decay=0.99, network_cfg: dict = None):
        super().__init__()
        self.online_net = build_module(**network_cfg)
        self.target_net = build_module(**network_cfg)
        for t_p, o_p in zip(self.target_net.parameters(), self.online_net.parameters()):
            t_p.data.copy_(o_p.data)
        set_requires_grad(self.target_net, False)
        self.criterion = nn.CrossEntropyLoss(ignore_index=255)

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

    @torch.no_grad()
    def forward_pseudo_label(self, image):
        # generate pseudo labels first
        image = self.normalize(image)
        self.target_net.train()
        pred_u_teacher = self.target_net(image)["pred"]
        pred_u_teacher = resize_as(pred_u_teacher, image)
        pred_u_teacher = pred_u_teacher.softmax(dim=1)
        logits_u, label_u = torch.max(pred_u_teacher, dim=1)
        entropy = -torch.sum(pred_u_teacher * torch.log(pred_u_teacher + 1e-10), dim=1)
        # apply strong data augmentation: cutout, cutmix, or classmix
        if np.random.uniform(0, 1) < 0.5:
            image, label_u, entropy = generate_unsup_data(image, label_u.clone(), entropy.clone(), mode='cutmix')
        return image, label_u, entropy

    def forward_train(self, image, target, image_u):
        image_u, target_u, entropy_u = self.forward_pseudo_label(image_u.clone())
        image = self.augment(image)
        image_u = self.strong_augment(image_u)
        image_all = torch.cat([image, image_u], dim=0)
        outputs = self.online_net(image_all)
        pred_all = outputs['pred']
        num_labeled, num_unlabeled = image.size(0), image_u.size(0)
        pred, pred_u = torch.split(pred_all, [num_labeled, num_unlabeled], dim=0)

        # supervised loss
        sup_loss = self.criterion(resize_as(pred, image), target)

        # unsupervised loss
        pseudo_loss = pseudo_segmentation_loss(resize_as(pred_u, target_u), target_u.clone(), 80, entropy_u)

        # moving avg
        update_model_moving_average(self.ema_decay, self.target_net, self.online_net)
        return dict(seg_loss=sup_loss, reg_loss=pseudo_loss)

    def forward_test(self, input):
        x = self.normalize(input)
        pred = self.online_net(x)
        return pred

    def forward(self, input, target=None, input_u=None):
        if not self.training:
            return self.forward_test(input)
        loss = self.forward_train(input, target, input_u)
        return loss

    def parameter_groups(self):
        groups = ([], [])
        backbone = [self.online_net.backbone]
        newly_added = [self.online_net.decoder,
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
