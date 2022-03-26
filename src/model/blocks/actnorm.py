import torch
import torch.nn as nn


class ActNorm(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.loc = nn.Parameter(torch.zeros(1, channels, 1, 1))
        self.scale = nn.Parameter(torch.ones(1, channels, 1, 1))
        self.initialized = False

    @torch.no_grad()
    def initialize(self, x):
        channels = x.size(1)

        flatten = x.permute(1, 0, 2, 3).contiguous().view(x.shape[1], -1)
        mean = flatten.mean(1).view(1, channels, 1, 1)
        std = flatten.std(1).view(1, channels, 1, 1)

        self.loc.data.copy_(-mean)
        self.scale.data.copy_(1 / (std + 1e-6))

    def forward(self, x):
        bs, channels, height, width = x.size()

        if not self.initialized:
            self.initialize(x)
            self.initialized = True

        log_abs = torch.log(torch.abs(self.scale))
        log_det = height * width * log_abs.sum()

        return self.scale * (x + self.loc), log_det

    def reverse(self, x):
        return x / self.scale - self.loc
