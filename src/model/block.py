import torch
import torch.nn as nn
from .utils import gaussian_log_p, gaussian_sample
from .blocks import ZeroConv2d
from .flow import Flow


class Block(nn.Module):
    def __init__(self, in_channels: int, n_flow: int, split: bool = True, affine: bool = True):
        super(Block, self).__init__()

        squeeze_dim = in_channels * 4

        self.flows = nn.ModuleList()
        for i in range(n_flow):
            self.flows.append(Flow(squeeze_dim, affine=affine))

        self.split = split

        if self.split:
            self.prior = ZeroConv2d(in_channels * 2, in_channels * 4)
        else:
            self.prior = ZeroConv2d(in_channels * 4, in_channels * 8)

    def forward(self, x):
        bs, channels, height, width = x.size()

        squeezed = x.view(bs, channels, height // 2, 2, width // 2, 2)
        squeezed = squeezed.permute(0, 1, 3, 5, 2, 4)
        x = squeezed.contiguous().view(bs, channels * 4, height // 2, width // 2)

        log_det = 0

        for flow in self.flows:
            x, curr_log_det = flow(x)
            log_det += curr_log_det

        if self.split:
            x, z_new = x.chunk(2, 1)
            mean, log_sd = self.prior(x).chunk(2, 1)
            log_p = gaussian_log_p(z_new, mean, log_sd)
            log_p = log_p.view(bs, -1).sum(1)
        else:
            zero = torch.zeros_like(x)
            mean, log_sd = self.prior(zero).chunk(2, 1)
            log_p = gaussian_log_p(x, mean, log_sd)
            log_p = log_p.view(bs, -1).sum(1)
            z_new = x

        return x, log_det, log_p, z_new

    def reverse(self, x, eps: float = None, reconstruct: bool = False):
        if reconstruct:
            if self.split:
                x = torch.cat([x, eps], 1)
            else:
                x = eps
        else:
            if self.split:
                mean, log_sd = self.prior(x).chunk(2, 1)
                z = gaussian_sample(eps, mean, log_sd)
                x = torch.cat((x, z), 1)
            else:
                zero = torch.zeros_like(x)
                mean, log_sd = self.prior(zero).chunk(2, 1)
                x = gaussian_sample(eps, mean, log_sd)

        for flow in self.flows[::-1]:
            x = flow.reverse(x)

        bs, channels, height, width = x.size()

        unsqueezed = x.view(bs, channels // 4, 2, 2, height, width)
        unsqueezed = unsqueezed.permute(0, 1, 4, 2, 5, 3)
        unsqueezed = unsqueezed.contiguous().view(
            bs, channels // 4, height * 2, width * 2
        )

        return unsqueezed
