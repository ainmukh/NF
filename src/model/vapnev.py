import torch
import torch.nn as nn
from .glow_vapnev import Glow
from .utils import gaussian_log_p, gaussian_sample


class VAPNEV(nn.Module):
    def __init__(self, config):
        super(VAPNEV, self).__init__()
        # ENCODER
        self.encoder = []
        start_channels = 32
        self.encoder.append(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=7, padding=3)
        )
        for i in range(4):
            in_channels = start_channels * (2 ** i)
            out_channels = in_channels * 2
            self.encoder.append(nn.Sequential(
                nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(in_channels),
                nn.LeakyReLU(),
                nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.LeakyReLU(),
            ))
        self.encoder.append(nn.Flatten())
        self.encoder = nn.Sequential(*self.encoder)
        self.mean = nn.Linear(start_channels * (2 ** 4) * 4 * 4, 256)
        self.log_var = nn.Linear(start_channels * (2 ** 4) * 4 * 4, 256)
        # ENCODER

        # DECODER
        self.decoder = []
        self.decoder_map = nn.Linear(256, 512 * 4 * 4)
        start_channels = 512
        for i in range(4):
            in_channels = start_channels // (2 ** i)
            out_channels = in_channels // 2
            self.decoder.append(nn.Sequential(
                nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(in_channels),
                nn.LeakyReLU(),
                nn.Upsample(scale_factor=2, mode='bilinear'),
                nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.LeakyReLU(),
            ))
        self.decoder.append(
            nn.Conv2d(in_channels=start_channels // (2 ** 4), out_channels=3, kernel_size=7, padding=3)
        )
        self.decoder = nn.Sequential(*self.decoder)
        # DECODER

        # GLOW
        self.glow = Glow(config)
        # GLOW

        self.hidden = config.z_dim
        self.device = config.device

    def forward(self, x):
        z = self.encoder(x)
        print('Z SIZE =', z.size())
        mean, log_sd = self.decoder(z)
        y, log_det = self.glow(x)
        log_p = gaussian_log_p(y, mean, log_sd)
        return y, log_p, log_det

    @torch.no_grad()
    def reverse(self, x, t=0.5):
        z, _, _ = self.encoder(x)
        mean, log_sd = self.decoder(z)
        y = gaussian_sample(torch.randn_like(mean), mean, log_sd)
        result = self.cond_glow.reverse(y)
        return result

    @torch.no_grad()
    def sample(self, n, t=0.5):
        z = torch.randn(n, self.hidden).to(self.device) * t
        print('SAMPLE Z =', z.size())
        mean, log_sd = self.decoder(z)
        y = gaussian_sample(torch.randn_like(mean), mean, log_sd)
        return self.glow.reverse(y)