import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.mods.conv_module import ConvModule


class ASPP(nn.Module):
    def __init__(self, in_channels, inner_channels=256, out_channels=256, dilations=(1, 12, 24, 36), dropout_p=0.1,
                 norm_layer=nn.BatchNorm2d, relu_inplace=False):
        super(ASPP, self).__init__()

        self.aspp1 = ConvModule(in_channels, inner_channels, 1, padding=0, dilation=dilations[0],
                                norm_layer=norm_layer, relu_inplace=relu_inplace)
        self.aspp2 = ConvModule(in_channels, inner_channels, 3, padding=dilations[1], dilation=dilations[1],
                                norm_layer=norm_layer, relu_inplace=relu_inplace)
        self.aspp3 = ConvModule(in_channels, inner_channels, 3, padding=dilations[2], dilation=dilations[2],
                                norm_layer=norm_layer, relu_inplace=relu_inplace)
        self.aspp4 = ConvModule(in_channels, inner_channels, 3, padding=dilations[3], dilation=dilations[3],
                                norm_layer=norm_layer, relu_inplace=relu_inplace)

        self.global_avg_pool = nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)),
                                             ConvModule(in_channels, inner_channels, 1,
                                                        norm_layer=norm_layer, relu_inplace=relu_inplace)
                                             )

        self.bottleneck = ConvModule(5 * inner_channels, out_channels, 1,
                                     norm_layer=norm_layer, relu_inplace=relu_inplace)
        self.dropout = nn.Dropout2d(dropout_p)

    def forward(self, x):
        x1 = self.aspp1(x)
        x2 = self.aspp2(x)
        x3 = self.aspp3(x)
        x4 = self.aspp4(x)
        x5 = self.global_avg_pool(x)
        x5 = F.interpolate(x5, size=x4.size()[2:], mode='bilinear', align_corners=True)
        x = torch.cat((x1, x2, x3, x4, x5), dim=1)
        x = self.bottleneck(x)
        x = self.dropout(x)
        return x


class ASPPDecoder(nn.Module):
    def __init__(self, in_channels=(256, 2048), inner_channels=256, out_channels=256, dilations=(1, 12, 24, 36),
                 norm_layer=nn.BatchNorm2d, relu_inplace=True):
        super(ASPPDecoder, self).__init__()
        assert isinstance(dilations, (list, tuple))

        self.aspp = ASPP(in_channels[-1], inner_channels, out_channels, dilations=dilations,
                         norm_layer=norm_layer, relu_inplace=relu_inplace)

        self.conv_low_level = ConvModule(in_channels[0], inner_channels, 1,
                                         norm_layer=norm_layer, relu_inplace=relu_inplace)

    def forward(self, f_list):
        f = self.aspp(f_list[-1])
        f_low = f_list[0]
        f_low = self.conv_low_level(f_low)
        f = F.interpolate(f, size=f_low.shape[-2:], mode='bilinear', align_corners=True)
        f = torch.cat((f, f_low), dim=1)
        return f


if __name__ == '__main__':
    mod = ASPPDecoder((256, 256))
    x = torch.randn(2, 256, 64, 64)
    x1 = torch.randn(2, 256, 128, 128)
    o = mod((x1, x))
    print(o.shape)

