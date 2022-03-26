import torch
import torch.nn as nn


class VAE(nn.Module):
    def __init__(self, config):
        super().__init__()

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

        self.device = config.device

    def encode(self, x):
        latent = self.encoder(x)
        mean, log_var = self.mean(latent), self.log_var(latent)
        
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)

        z = mean + std * eps

        return z, mean, log_var

    def decode(self, z):
        latent = self.decoder_map(z).reshape(-1, 512, 4, 4)
        x = self.decoder(latent)
        
        return x

    def forward(self, x):
        z, mean, log_var = self.encode(x)
        x = self.decode(z)

        return x, mean, log_var

    @torch.no_grad()
    def sample(self, n_samples):
        samples = torch.randn(n_samples, 256).to(self.device)
        sample_res = self.decode(samples)

        return sample_res
