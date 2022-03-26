import torch
import torch.nn as nn
import torch.nn.functional as F


class InvConv2dLU(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        weight = torch.randn(channels, channels)
        q, r = torch.linalg.qr(weight)
        w_p, w_l, w_u = torch.lu_unpack(*q.lu())

        w_s = w_u.diag()
        w_u = w_u.triu(1)
        u_mask = torch.ones_like(w_u).triu(1)
        l_mask = u_mask.T

        self.register_buffer('w_p', w_p)
        self.register_buffer('u_mask', u_mask)
        self.register_buffer('l_mask', l_mask)
        self.register_buffer('s_sign', torch.sign(w_s))
        self.register_buffer('l_eye', torch.eye(l_mask.size(0)))

        self.w_l = nn.Parameter(w_l)
        self.w_s = nn.Parameter(torch.log(torch.abs(w_s)))
        self.w_u = nn.Parameter(w_u)

    def calc_weight(self):
        weight = (
                self.w_p
                @ (self.w_l * self.l_mask + self.l_eye)
                @ ((self.w_u * self.u_mask) + torch.diag(self.s_sign * torch.exp(self.w_s)))
        )

        return weight.unsqueeze(2).unsqueeze(3)

    def forward(self, x):
        bs, channels, height, width = x.size()

        weight = self.calc_weight()

        x = F.conv2d(x, weight)
        log_det = height * width * self.w_s.sum()

        return x, log_det

    def reverse(self, x):
        weight = self.calc_weight()

        return F.conv2d(
            x, weight.squeeze().inverse().unsqueeze(2).unsqueeze(3)
        )
