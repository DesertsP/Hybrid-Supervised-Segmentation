import torch
import torch.nn as nn
from utils import build_module
from models.mods.ops import set_requires_grad, resize_as
from kornia import augmentation as augs
from models.mods.ops import update_model_moving_average


class MeanTeacherLearner(nn.Module):
    def __init__(self, ema_decay=0.99, use_augment=False, use_pseudo_loss=False, network_cfg: dict = None):
        super().__init__()
        self.online_net = build_module(**network_cfg)
        self.target_net = build_module(**network_cfg)
        for t_p, o_p in zip(self.target_net.parameters(), self.online_net.parameters()):
            t_p.data.copy_(o_p.data)
        set_requires_grad(self.target_net, False)
        self.criterion = nn.CrossEntropyLoss(ignore_index=255)
        self.consistency_criterion = nn.MSELoss(reduction='mean')
        self.use_pseudo_loss = use_pseudo_loss
        self.ema_decay = ema_decay

        self.normalize = augs.Normalize(mean=torch.tensor((0.485, 0.456, 0.406)),
                                        std=torch.tensor((0.229, 0.224, 0.225)))
        DEFAULT_AUG = nn.Sequential(
            augs.ColorJitter(0.5, 0.5, 0.5, 0.1, p=0.5),
            # augs.RandomGaussianBlur((3, 3), (1.5, 1.5), p=0.5),
            # augs.RandomSolarize(p=0.1),
            self.normalize
        )
        self.augment = DEFAULT_AUG if use_augment else self.normalize

    def consistency_loss(self, pred_online, pred_target):
        return self.consistency_criterion(pred_online.softmax(dim=1), pred_target.softmax(dim=1).detach())

    def pseudo_loss(self, pred_online, pred_target):
        pseudo_label = torch.argmax(pred_target, dim=1).long()
        return self.criterion(pred_online, pseudo_label)

    def forward_train(self, input, target=None):
        x_online = self.augment(input)
        x_target = self.normalize(input)
        pred_online = self.online_net(x_online)['pred']
        with torch.no_grad():
            pred_target = self.target_net(x_target)['pred']
        # mean teacher loss
        if self.use_pseudo_loss:
            reg_loss = self.pseudo_loss(pred_online, pred_target)
        else:
            reg_loss = self.consistency_loss(pred_online, pred_target)

        pred_online = resize_as(pred_online, x_online)
        # seg loss
        if target is not None:
            seg_loss = self.criterion(pred_online, target)
        else:
            seg_loss = 0.0
        return dict(seg_loss=seg_loss, reg_loss=reg_loss)

    def forward_test(self, input):
        x = self.normalize(input)
        pred = self.online_net(x)
        return pred

    def forward(self, input, target=None, input_u=None):
        if not self.training:
            return self.forward_test(input)
        loss = self.forward_train(input, target)
        if input_u is not None:
            loss['reg_loss'] += self.forward_train(input_u)['reg_loss']
        update_model_moving_average(self.ema_decay, self.target_net, self.online_net)
        return loss

    def parameter_groups(self):
        groups = ([], [])
        backbone = [self.online_net.backbone]
        newly_added = [self.online_net.decoder,
                       self.online_net.classification_loss, ]
        for module in backbone:
            for p in module.parameters():
                groups[0].append(p)
        for module in newly_added:
            for p in module.parameters():
                groups[1].append(p)
        assert len(list(self.parameters())) == 2 * sum([len(g) for g in groups])
        return groups
