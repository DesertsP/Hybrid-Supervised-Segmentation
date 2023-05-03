import torch
import torch.nn as nn
import torch.nn.functional as F
from models.mods.conv_module import ConvModule
from utils import build_module, get_module


class DeepLab(nn.Module):
    def __init__(self, num_classes, backbone_cfg: dict, decoder_cfg: dict, norm_cfg: dict, aux_output=False):
        super(DeepLab, self).__init__()
        norm_layer = get_module(**norm_cfg)
        backbone_cfg.update({'norm_layer': norm_layer})
        decoder_cfg.update({'norm_layer': norm_layer})
        self.backbone = build_module(**backbone_cfg)
        self.decoder = build_module(**decoder_cfg)
        self.aux_output = aux_output

        self.projector = nn.Sequential(
            ConvModule(512, 256, 3, padding=1, bias=True, norm_layer=norm_layer, relu_inplace=True),
            nn.Dropout2d(0.1),
            ConvModule(256, 256, 3, padding=1, bias=True, norm_layer=norm_layer, relu_inplace=True),
            nn.Dropout2d(0.1)
        )
        self.classifier = nn.Conv2d(256, num_classes, kernel_size=1, stride=1, padding=0, bias=True)

        # for deep supervision
        if aux_output:
            self.aux_classifier = nn.Sequential(
                ConvModule(1024, 256, 3, padding=1, bias=True, norm_layer=norm_layer, relu_inplace=True),
                nn.Dropout2d(0.1),
                nn.Conv2d(256, num_classes, kernel_size=1, stride=1, padding=0, bias=True)
            )
            self._init_weight(self.aux_classifier)
        self._init_weight(self.projector)
        self._init_weight(self.decoder)
        self._init_weight(self.classifier)

    def forward(self, x):
        feats = self.backbone(x)
        x = self.decoder(feats)
        x = self.projector(x)
        pred = self.classifier(x)
        output = {'pred': pred}
        if self.aux_output:
            output['aux'] = self.aux_classifier(feats[-2])
        return output

    @staticmethod
    def _init_weight(module):
        for m in module.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.SyncBatchNorm):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


if __name__ == "__main__":
    model = DeepLab(20,
                    {'name': 'models.backbones.resnet.resnet101', 'pretrained': None,
                     'replace_stride_with_dilation': [False, True, True], 'zero_init_residual': True},
                    {'name': 'models.mods.aspp.ASPPDecoder'},
                    {'name': 'torch.nn.BatchNorm2d'})
    model.eval()
    input = torch.rand(2, 3, 513, 513)
    output = model(input)
    print(output.size())
