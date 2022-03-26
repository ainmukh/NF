import torch
import torch.nn as nn
import torch.nn.functional as F
from .zeroconv import ZeroConv2d


class AffineCoupling(nn.Module):
    def __init__(self, in_channels: int, filter_size: int = 512, affine: bool = True):
        super().__init__()
        self.affine = affine

        self.net = nn.Sequential(
            nn.Conv2d(in_channels // 2, filter_size, 3, padding=1),
            nn.ReLU(True),
            nn.Conv2d(filter_size, filter_size, 1),
            nn.ReLU(True),
            ZeroConv2d(filter_size, (in_channels // 2, in_channels)[self.affine])
        )

        self.net[0].weight.data.normal_(0, 0.05)
        self.net[0].bias.data.zero_()

        self.net[2].weight.data.normal_(0, 0.05)
        self.net[2].bias.data.zero_()

    def forward(self, x):
        in_a, in_b = x.chunk(2, 1)
        log_det = None

        if self.affine:
            log_s, t = self.net(in_a).chunk(2, 1)
            s = F.sigmoid(log_s + 2)
            out_b = (in_b + t) * s

            log_det = torch.sum(torch.log(s).view(x.size(0), -1), 1)
        else:
            net_out = self.net(in_a)
            out_b = in_b + net_out

        return torch.cat((in_a, out_b), 1), log_det

    def reverse(self, x):
        out_a, out_b = x.chunk(2, 1)

        if self.affine:
            log_s, t = self.net(out_a).chunk(2, 1)
            s = F.sigmoid(log_s + 2)
            in_b = out_b / s - t
        else:
            net_out = self.net(out_a)
            in_b = out_b - net_out

        return torch.cat((out_a, in_b), 1)
