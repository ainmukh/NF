import torch
import torch.nn as nn
import torch.nn.functional as F


class ZeroConv2d(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super(ZeroConv2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, 3)
        self.conv.weight.data.zero_()
        self.conv.bias.data.zero_()
        self.scale = nn.Parameter(torch.zeros(1, out_channels, 1, 1))

    def forward(self, x):
        x = F.pad(x, [1, 1, 1, 1], value=1)
        x = self.conv(x)
        x = x * torch.exp(self.scale * 3)
