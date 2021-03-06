import torch
import torch.nn as nn
import torch.nn.functional as F
from .block_vapnev import Block
# from .block import Block


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
            n_channel *= 4
        self.blocks.append(
            Block(n_channel, config.n_flow, split=False, affine=config.affine)
        )
        self.device = config.device

    def forward(self, x):
        log_det = 0

        for block in self.blocks:
            x, det = block(x)
            log_det += det

        return x, log_det

    def reverse(self, x):
        for i, block in enumerate(self.blocks[::-1]):
            x = block.reverse(x)
        return x