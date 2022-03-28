import itertools
import torch
import torch.nn as nn
from torch.distributions import MultivariateNormal
from .utils import gaussian_log_p, gaussian_sample
from .seminar_flows import ActNorm, AffineHalfFlow, NormalizingFlowModel
from .vapnev_glow import Encoder, Decoder
from .blocks import AffineCoupling


class VAPNEV(nn.Module):
    def __init__(self):
        super(VAPNEV, self).__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()

        prior = MultivariateNormal(torch.zeros(2), torch.eye(2))
        # flows = [AffineHalfFlow(dim=2, parity=i % 2) for i in range(9)]
        flows = [AffineCoupling(3) for i in range(9)]
        norms = [ActNorm(dim=2) for _ in flows]
        flows = list(itertools.chain(*zip(norms, flows)))
        self.nvp = NormalizingFlowModel(prior, flows)

    def forward(self, x):
        z, _, _ = self.encoder(x)
        mean, log_sd = self.decoder(z)
        y, log_det = self.nvp(x)
        log_p = gaussian_log_p(y, mean, log_sd)
        return y, log_p, log_det

    @torch.no_grad()
    def reverse(self, x):
        z, _, _ = self.encoder(x)
        mean, log_sd = self.decoder(z)
        y = gaussian_sample(torch.randn_like(mean), mean, log_sd)
        result = self.nvp.reverse(y)
        return result
