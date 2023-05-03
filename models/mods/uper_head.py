import torch
import torch.nn as nn

from models.mods.ops import resize
from models.mods.psp_head import PPM


class ConvModule(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, padding=0, relu_inplace=False, bn_eps=1e-5,
                 bn_momentum=0.1):
        super().__init__()
        self.add_module('conv', nn.Conv2d(in_channels, out_channels, kernel_size,
                                          padding=padding, stride=stride, bias=False))
        self.add_module('norm', nn.BatchNorm2d(out_channels, momentum=bn_momentum, eps=bn_eps))
        self.add_module('relu', nn.ReLU(inplace=relu_inplace))


class UPerHead(nn.Module):
    """Unified Perceptual Parsing for Scene Understanding.

    This head is the implementation of `UPerNet
    <https://arxiv.org/abs/1807.10221>`_.

    Args:
        pool_scales (tuple[int]): Pooling scales used in Pooling Pyramid
            Module applied on the last feature. Default: (1, 2, 3, 6).
    """

    def __init__(self, in_channels, channels, pool_scales=(1, 2, 3, 6)):
        super(UPerHead, self).__init__()
        # PSP Module
        self.psp_modules = PPM(
            pool_scales,
            in_channels[-1],
            channels,
            align_corners=False)
        self.bottleneck = ConvModule(in_channels[-1] + len(pool_scales) * channels, channels,
                                     kernel_size=3, padding=1)
        # FPN Module
        self.lateral_convs = nn.ModuleList()
        self.fpn_convs = nn.ModuleList()
        for n_c in in_channels[:-1]:  # skip the top layer
            l_conv = ConvModule(
                n_c,
                channels,
                1)
            fpn_conv = ConvModule(
                channels,
                channels,
                3,
                padding=1)
            self.lateral_convs.append(l_conv)
            self.fpn_convs.append(fpn_conv)

        self.fpn_bottleneck = ConvModule(
            len(in_channels) * channels,
            channels,
            3,
            padding=1)

    def psp_forward(self, inputs):
        """Forward function of PSP module."""
        x = inputs[-1]
        psp_outs = [x]
        psp_outs.extend(self.psp_modules(x))
        psp_outs = torch.cat(psp_outs, dim=1)
        output = self.bottleneck(psp_outs)

        return output

    def forward(self, inputs):
        """Forward function."""

        # build laterals
        laterals = [
            lateral_conv(inputs[i])
            for i, lateral_conv in enumerate(self.lateral_convs)
        ]

        laterals.append(self.psp_forward(inputs))

        # build top-down path
        used_backbone_levels = len(laterals)
        for i in range(used_backbone_levels - 1, 0, -1):
            prev_shape = laterals[i - 1].shape[2:]
            laterals[i - 1] += resize(
                laterals[i],
                size=prev_shape,
                mode='bilinear',
                align_corners=False)

        # build outputs
        fpn_outs = [
            self.fpn_convs[i](laterals[i])
            for i in range(used_backbone_levels - 1)
        ]
        # append psp feature
        fpn_outs.append(laterals[-1])

        for i in range(used_backbone_levels - 1, 0, -1):
            fpn_outs[i] = resize(
                fpn_outs[i],
                size=fpn_outs[0].shape[2:],
                mode='bilinear',
                align_corners=False)
        fpn_outs = torch.cat(fpn_outs, dim=1)
        output = self.fpn_bottleneck(fpn_outs)
        return output
