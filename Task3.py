import itertools
import torch
import torch.nn as nn
from src.model.vapnev_glow import Encoder, Decoder
# from src.model.blocks import ActNorm
from src.model.seminar_flows import NormalizingFlowModel
from src.model.utils import gaussian_sample, gaussian_log_p


class ActNorm(nn.Module):
    def __init__(self, channels):
        super(ActNorm, self).__init__()

        self.log_scale = nn.Parameter(torch.randn(channels))
        self.loc = nn.Parameter(torch.zeros(channels))

        self.initialized = False

    def initialize(self, x):
        mean = x.mean(0)
        std = x.std(0) + 1e-4

        self.loc.data.copy_(-mean)
        self.log_scale.copy_(-torch.log(std))

    def forward(self, x):
        if not self.initialized:
            self.initialize(x)
            self.initialized = True

        x = torch.exp(self.log_scale[None, :]) * x + self.loc
        log_det = self.log_scale.sum()
        return x, log_det

    def reverse(self, x):
        return torch.exp(-self.log_scale[None, :]) * (x - self.loc)


class InvNonLin(nn.Module):
    def __init__(self):
        super(InvNonLin, self).__init__()

    def forward(self, x):
        mask = (torch.abs(x) < 1).float()
        power = 3 * mask + 1 * (1 - mask)
        x = torch.sign(x) * ((x * torch.sign(x)) ** power)

        log_det = torch.log((3 * (x ** 2)) * mask + (1 - mask)).sum(-1)

        return x, log_det

    def reverse(self, x):
        mask = (torch.abs(x) < 1).float()
        power = (1 / 3) * mask + 1 * (1 - mask)

        x = torch.sign(x) * ((x * torch.sign(x)) ** power)

        return x


class RealNVP(nn.Module):
    def __init__(self, in_size, mask, hidden=64):
        super(RealNVP, self).__init__()

        self.mask = mask

        self.t = nn.Sequential(
            nn.Linear(in_size, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, in_size)
        )
        self.s = nn.Sequential(
            nn.Linear(in_size, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, in_size),
        )

    def forward(self, x):
        t = self.t(x * self.mask[None, :])
        s = torch.tanh(self.s(x * self.mask[None, :]))

        x = (x * torch.exp(s) + t) * (1 - self.mask[None, :]) + x * self.mask[None, :]
        log_det = (s * (1 - self.mask[None, :])).sum(-1)
        return x, log_det

    def reverse(self, x):
        t = self.t(x * self.mask[None, :])
        s = torch.tanh(self.s(x * self.mask[None, :]))

        x = ((x - t) * torch.exp(-s)) * (1 - self.mask[None, :]) + x * self.mask[None, :]
        return x


class Task3(nn.Module):
    def __init__(self, device, latent_size=256, lambda_kl=0.001):
        super(Task3, self).__init__()

        pic_size = 3 * 64 * 64
        self.latent_size = latent_size
        self.encoder = Encoder()
        self.decoder = Decoder()

        prior = torch.distributions.MultivariateNormal(torch.zeros(pic_size).cuda(), torch.eye(pic_size).cuda())
        flows = []
        for i in range(6):
            flows.append(RealNVP(in_size=pic_size, mask=torch.randint(0, 2, (pic_size,)).cuda()))
            flows.append(ActNorm(channels=pic_size))

        self.prior = NormalizingFlowModel(prior, flows)

        self.k = 0
        self.lambda_kl = lambda_kl
        self.device = device

    def encode(self, x):
        return self.encoder(x)

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        bs, channels, _, _ = x.size()
        z, _, _ = self.encoder(x)
        rec, mean, log_sd = self.decoder(z)
        y, log_det = self.prior(x.view(bs, -1))
        log_p = gaussian_log_p(y[-1], mean, log_sd)
        return rec, log_p, log_det

    @torch.no_grad()
    def sample(self, n):
        z = torch.randn(n, 256).to(self.device) / 2
        _, mean, log_std = self.decoder(z)
        # bs = mean.size(0)
        # mean = mean.view(bs, 3, 64, 64)
        # log_std = log_std.view(bs, 3, 64, 64)
        return gaussian_sample(torch.randn_like(mean), mean, log_std)
        # return mean + torch.exp(log_std / 2) * eps  делит на 2 почему-то

    @torch.no_grad()
    def reverse(self, x):
        z, _, _ = self.encoder(x)
        _, mean, log_std = self.decoder(z)
        y = gaussian_sample(torch.randn_like(mean), mean, log_std)
        result = self.prior.reverse(y)
        result = result[-1].view(z.size(0), 3, 64, 64)
        return result

    # def custom_loss(self, x, rec_x, z, mu, log_sigma):
    #
    #     h - log_sd - 0.5 * (x - mean) ** 2 / torch.exp(2 * log_sd)
    #
    #     h = -self.latent_size * torch.log(2 * np.pi) / 2
    #     h = h - ((z - mu) ** 2 / (torch.exp(log_sigma) + 1e-2) + log_sigma).sum(dim=1) / 2
    #
    #     p_z = self.prior.log_prob(z)  # todo add losses to watch
    #     KL = torch.mean(h - p_z)
    #
    #     recon_loss = self.criterion(x, rec_x)
    #
    #     l2_loss = KL * self.KL_weight + recon_loss
    #
    #     return l2_loss

    # def compute_loss(self, x, return_rec=False):
    #
    #     _, mu, log_sigma = self.encode(x)
    #     z = self.sample_z(mu, log_sigma)
    #     rec_x = self.decode(z)
    #
    #     l2_loss = self.custom_loss(x, rec_x, z, mu, log_sigma)
    #
    #     if return_rec:
    #         return l2_loss, mu, log_sigma, rec_x
    #     else:
    #         return l2_loss, mu, log_sigma
