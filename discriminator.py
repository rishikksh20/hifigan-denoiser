import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.nn import Conv1d, ConvTranspose1d, AvgPool1d, Conv2d
from torch.nn.utils import weight_norm, remove_weight_norm, spectral_norm



class MelganDiscriminator(torch.nn.Module):
    def __init__(self, use_spectral_norm=False):
        super(MelganDiscriminator, self).__init__()
        norm_f = weight_norm if use_spectral_norm == False else spectral_norm
        self.convs = nn.ModuleList([
            norm_f(Conv1d(1, 128, 15, 1, padding=7)),
            norm_f(Conv1d(128, 128, 41, 2, groups=4, padding=20)),
            norm_f(Conv1d(128, 256, 41, 2, groups=16, padding=20)),
            norm_f(Conv1d(256, 512, 41, 4, groups=16, padding=20)),
            norm_f(Conv1d(512, 1024, 41, 4, groups=16, padding=20)),
            norm_f(Conv1d(1024, 1024, 41, 1, groups=16, padding=20)),
            norm_f(Conv1d(1024, 1024, 5, 1, padding=2)),
        ])
        self.conv_post = norm_f(Conv1d(1024, 1, 3, 1, padding=1))

    def forward(self, x):
        fmap = []
        for l in self.convs:
            x = l(x)
            x = F.leaky_relu(x, 0.1)
            fmap.append(x)
        x = self.conv_post(x)
        fmap.append(x)
        x = torch.flatten(x, 1, -1)

        return x, fmap


class MultiScaleDiscriminator(torch.nn.Module):
    def __init__(self):
        super(MultiScaleDiscriminator, self).__init__()
        self.discriminators = nn.ModuleList([
            MelganDiscriminator(use_spectral_norm=True),
            MelganDiscriminator(),
            MelganDiscriminator(),
        ])
        self.meanpools = nn.ModuleList([
            AvgPool1d(4, 2, padding=2),
            AvgPool1d(4, 2, padding=2)
        ])

    def forward(self, y, y_hat):
        y_d_rs = []
        y_d_gs = []
        fmap_rs = []
        fmap_gs = []
        for i, d in enumerate(self.discriminators):
            if i != 0:
                y = self.meanpools[i-1](y)
                y_hat = self.meanpools[i-1](y_hat)
            y_d_r, fmap_r = d(y)
            y_d_g, fmap_g = d(y_hat)
            y_d_rs.append(y_d_r)
            fmap_rs.append(fmap_r)
            y_d_gs.append(y_d_g)
            fmap_gs.append(fmap_g)

        return y_d_rs, y_d_gs, fmap_rs, fmap_gs


class Down2d(nn.Module):
    """docstring for Down2d."""

    def __init__(self, in_channel, out_channel, kernel, stride, padding):
        super(Down2d, self).__init__()

        self.c1 = nn.Conv2d(in_channel, out_channel, kernel_size=kernel, stride=stride, padding=padding)
        self.n1 = nn.InstanceNorm2d(out_channel)
        self.c2 = nn.Conv2d(in_channel, out_channel, kernel_size=kernel, stride=stride, padding=padding)
        self.n2 = nn.InstanceNorm2d(out_channel)

    def forward(self, x):
        x1 = self.c1(x)
        x1 = self.n1(x1)

        x2 = self.c2(x)
        x2 = self.n2(x2)

        x3 = x1 * torch.sigmoid(x2)

        return x3



class SpecDiscriminator(nn.Module):
    """docstring for Discriminator."""

    def __init__(self):
        super(SpecDiscriminator, self).__init__()

        self.discriminators = nn.ModuleList([
            Down2d(1, 32, (3, 9), (1, 2), (1, 4)),
            Down2d(32, 32, (3, 8), (1, 2), (1, 3)),
            Down2d(32, 32, (3, 8), (1, 2), (1, 3)),
            Down2d(32, 32, (3, 6), (1, 2), (1, 2)),
        ])
        self.conv = nn.Conv2d(32, 1, (32, 5), (32, 1), (0, 2))
        self.pool = nn.AvgPool2d((1, 2))

    def forward(self, y, y_hat):

        y_d_rs = []
        y_d_gs = []
        fmap_r = []
        fmap_g = []
        fmap_rs = []
        fmap_gs = []
        y = y.unsqueeze(1)
        y_hat = y_hat.unsqueeze(1)
        for i, d in enumerate(self.discriminators):
            y = d(y)
            y_hat = d(y_hat)
            fmap_r.append(y)
            fmap_g.append(y_hat)

        y = self.conv(y)
        fmap_r.append(y)
        y = self.pool(y)
        y_d_rs.append(torch.flatten(y, 1, -1))

        y_hat = self.conv(y_hat)
        fmap_g.append(y_hat)

        y_hat = self.pool(y_hat)
        y_d_gs.append(torch.flatten(y_hat, 1, -1))

        fmap_rs.append(fmap_r)
        fmap_gs.append(fmap_g)

        return y_d_rs, y_d_gs, fmap_rs, fmap_gs



def feature_loss(fmap_r, fmap_g):
    loss = 0
    for dr, dg in zip(fmap_r, fmap_g):
        for rl, gl in zip(dr, dg):
            loss += torch.mean(torch.abs(rl - gl))

    return loss*2


def discriminator_loss(disc_real_outputs, disc_generated_outputs):
    loss = 0
    r_losses = []
    g_losses = []
    for dr, dg in zip(disc_real_outputs, disc_generated_outputs):
        r_loss = torch.mean((1-dr)**2)
        g_loss = torch.mean(dg**2)
        loss += (r_loss + g_loss)
        r_losses.append(r_loss.item())
        g_losses.append(g_loss.item())

    return loss, r_losses, g_losses

