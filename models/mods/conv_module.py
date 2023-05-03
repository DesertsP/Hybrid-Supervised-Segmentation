import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.common_types import _size_2_t


class ConvModule(nn.Module):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: _size_2_t,
                 stride: _size_2_t = 1,
                 padding: _size_2_t = 0,
                 dilation: _size_2_t = 1,
                 groups: int = 1,
                 bias: bool = False,
                 norm_layer=nn.BatchNorm2d,
                 relu_inplace=False):
        super(ConvModule, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size,
                              stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        self.norm = norm_layer(out_channels)
        self.relu = nn.ReLU(relu_inplace)

    def forward(self, x):
        x = self.conv(x)
        x = self.norm(x)
        return self.relu(x)
