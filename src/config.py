import torch
from dataclasses import dataclass


@dataclass
class Config:
    in_channels: int = 3
    n_flows: int = 8
    n_blocks: int = 1
    lu: bool = True
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    lr: float = 2e-4
    batch: int = 8
    n_flow: int = 16
    n_block: int = 3
    conv_lu: bool = True
    affine: bool = True
    n_bits: int = 5
    img_size: int = 64
    temp: float = 0.7
    n_sample: int = 64
    z_dim: int = 256
