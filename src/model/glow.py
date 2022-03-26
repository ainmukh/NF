import torch
import torch.nn as nn
import torch.nn.functional as F
from .block import Block


class Glow(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.block = nn.ModuleList()

        n_channel = config.in_channels
        self.n_block = config.n_blocks
        self.n_flow = config.n_flows
        self.blocks = nn.ModuleList()
        for i in range(self.n_block - 1):
            self.blocks.append(
                Block(n_channel, config.n_flow, affine=config.affine)
            )
            n_channel *= 2
        self.blocks.append(
            Block(n_channel, config.n_flow, split=False, affine=config.affine)
        )
        self.device = config.device

    def init_with_data(self, batch):
        bs, channels, input_size, _ = batch.size()
        self.z_shapes = self.calc_z_shapes(
            channels, input_size, self.n_block
        )

    def calc_z_shapes(self, n_channels, input_size, n_blocks):
        z_shapes = []
        for i in range(n_blocks - 1):
            input_size //= 2
            n_channels *= 2
            z_shapes.append((n_channels, input_size, input_size))
        input_size //= 2
        z_shapes.append((n_channels * 4, input_size, input_size))
        return z_shapes

    def forward(self, x):
        log_p_sum = 0
        log_det = 0
        x = x
        z_outs = []

        for block in self.blocks:
            x, det, log_p, z_new = block(x)
            z_outs.append(z_new)
            log_det = log_det + det

            if log_p is not None:
                log_p_sum = log_p_sum + log_p

        return log_p_sum, log_det, z_outs

    def reverse(self, z_list, reconstruct=False):
        x = z_list[-1]
        for i, block in enumerate(self.blocks[::-1]):
            x = block.reverse(x, z_list[-(i + 1)], reconstruct=reconstruct)
        return x

    @torch.no_grad()
    def sample(self, n, t=0.5):
        z_sample = []
        for z in self.z_shapes:
            z_new = torch.randn(n, *z) * t
            z_sample.append(z_new.to(self.device))
        return self.reverse(z_sample)
