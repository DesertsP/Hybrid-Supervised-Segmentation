import torch
import torch.nn as nn
import torch.nn.functional as F
from models.mods.conv_module import ConvModule
from utils import build_module, get_module


class DeepLab(nn.Module):
    def __init__(self, num_classes, out_channels, backbone_cfg: dict, decoder_cfg: dict, norm_cfg: dict):
        super(DeepLab, self).__init__()
        norm_layer = get_module(**norm_cfg)
        backbone_cfg.update({'norm_layer': norm_layer})
        decoder_cfg.update({'norm_layer': norm_layer})
        self.backbone = build_module(**backbone_cfg)
        self.decoder = build_module(**decoder_cfg)

        self.classifier = nn.Sequential(
            ConvModule(512, 256, 3, padding=1, bias=True, norm_layer=norm_layer, relu_inplace=True),
            nn.Dropout2d(0.1),
            ConvModule(256, 256, 3, padding=1, bias=True, norm_layer=norm_layer, relu_inplace=True),
            nn.Dropout2d(0.1),
            nn.Conv2d(256, num_classes, kernel_size=1, stride=1, padding=0, bias=True),
        )
        self.projector = nn.Sequential(
            ConvModule(512, 256, 3, padding=1, bias=True, norm_layer=norm_layer, relu_inplace=True),
            nn.Dropout2d(0.1),
            ConvModule(256, 256, 3, padding=1, bias=True, norm_layer=norm_layer, relu_inplace=True),
            nn.Dropout2d(0.1),
            nn.Conv2d(256, out_channels, kernel_size=1, stride=1, padding=0, bias=True),
        )

    def forward(self, x):
        x = self.backbone(x)
        x = self.decoder(x)
        pred = self.classifier(x)
        proj = self.projector(x)
        return {'pred': pred, 'proj': proj}

    @staticmethod
    def _init_weight(module):
        for m in module.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.SyncBatchNorm):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


if __name__ == "__main__":
    model = DeepLab(20, 256,
                    {'name': 'models.backbones.resnet.resnet101', 'pretrained': '/home/pjw/pretrained/resnet101.pth',
                     'replace_stride_with_dilation': [False, True, True], 'zero_init_residual': True},
                    {'name': 'models.mods.aspp.ASPPDecoder'},
                    {'name': 'torch.nn.BatchNorm2d'})
    model.eval()
    input = torch.rand(2, 3, 513, 513)
    output = model(input)
    print(output.size())
