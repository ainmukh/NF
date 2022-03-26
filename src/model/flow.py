import torch.nn as nn
from .blocks import ActNorm, InvConv2dLU, AffineCoupling


class Flow(nn.Module):
    def __init__(self, in_channels: int, filter_size: int = 512, affine: bool = True):
        super(Flow, self).__init__()
        self.actnorm = ActNorm(in_channels)
        self.invconv = InvConv2dLU(in_channels)
        self.coupling = AffineCoupling(in_channels, filter_size, affine)

    def forward(self, x):
        x, log_det = self.actnorm(x)
        x, det1 = self.invconv(x)
        x, det2 = self.coupling(x)

        log_det = log_det + det1
        if det2 is not None:
            log_det = log_det + det2

        return x, log_det

    def reverse(self, x):
        x = self.coupling.reverse(x)
        x = self.invconv.reverse(x)
        x = self.actnorm.reverse(x)

        return x
