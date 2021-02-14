""" borrow from https://github.com/Sytronik/denoising-wavenet-pytorch/blob/master/model/dwavenet.py.
"""
import math

import torch
from torch import nn
from torch.nn import functional as F


class ResidualConv1dGLU(nn.Module):
    """Residual dilated conv1d + Gated linear unit
    Args:
        residual_channels (int): Residual input / output channels
        gate_channels (int): Gated activation channels.
        kernel_size (int): Kernel size of convolution layers.
        skip_out_channels (int): Skip connection channels. If None, set to same
          as ``residual_channels``.
        cin_channels (int): Local conditioning channels. If negative value is
          set, local conditioning is disabled.
        gin_channels (int): Global conditioning channels. If negative value is
          set, global conditioning is disabled.
        dropout (float): Dropout probability.
        padding (int): Padding for convolution layers. If None, proper padding
          is computed depends on dilation and kernel_size.
        dilation (int): Dilation factor.
    """

    def __init__(self, residual_channels, gate_channels, kernel_size,
                 skip_out_channels=None,
                 cin_channels=-1, gin_channels=-1,
                 dropout=1 - 0.95, padding=None, dilation=1,
                 bias=True, *args, **kwargs):
        super(ResidualConv1dGLU, self).__init__()
        self.dropout = dropout
        if skip_out_channels is None:
            skip_out_channels = residual_channels
        if padding is None:
            padding = (kernel_size - 1) // 2 * dilation   # For non causal dilated convolution

        self.conv = nn.Conv1d(residual_channels, gate_channels, kernel_size,
                              padding=padding, dilation=dilation,
                              bias=bias, *args, **kwargs)

        # conv output is split into two groups
        gate_out_channels = gate_channels // 2
        self.conv1x1_out = nn.Conv1d(gate_out_channels, residual_channels, 1, bias=bias)
        self.conv1x1_skip = nn.Conv1d(gate_out_channels, skip_out_channels, 1, bias=bias)

    def forward(self, x):
        """Forward
        Args:
            x (Tensor): B x C x T
            c (Tensor): B x C x T, Local conditioning features
            g (Tensor): B x C x T, Expanded global conditioning features
        Returns:
            Tensor: output
        """

        residual = x
        x = F.dropout(x, p=self.dropout, training=self.training)
        splitdim = 1
        x = self.conv(x)

        a, b = x.split(x.size(splitdim) // 2, dim=splitdim)


        x = torch.tanh(a) * torch.sigmoid(b)

        # For skip connection
        s = self.conv1x1_skip(x)

        # For residual connection
        x = self.conv1x1_out(x)

        x = (x + residual) * math.sqrt(0.5)
        return x, s


class Generator(nn.Module):
    def __init__(self, in_channels, out_channels=1, bias=False,
                 num_layers=20, num_stacks=2,
                 kernel_size=3,
                 residual_channels=128, gate_channels=128, skip_out_channels=128,
                 ):
        super().__init__()
        assert num_layers % num_stacks == 0
        num_layers_per_stack = num_layers // num_stacks
        # in_channels is 1 for RAW waveform otherwise quantize classes
        self.first_conv = nn.Conv1d(in_channels, residual_channels, 3, padding=1, bias=bias)

        self.conv_layers = nn.ModuleList()
        for n_layer in range(num_layers):
            dilation = 2**(n_layer % num_layers_per_stack)
            conv = ResidualConv1dGLU(
                residual_channels, gate_channels,
                skip_out_channels=skip_out_channels,
                kernel_size=kernel_size,
                bias=bias,
                dilation=dilation,
                dropout=1 - 0.95,
            )
            self.conv_layers.append(conv)

        self.last_conv_layers = nn.Sequential(
            nn.ReLU(True),
            nn.Conv1d(skip_out_channels, skip_out_channels, 1, bias=True),
            nn.ReLU(True),
            nn.Conv1d(skip_out_channels, out_channels, 1, bias=True),
        )

    def forward(self, x):
        x = self.first_conv(x)
        skips = 0
        for conv in self.conv_layers:
            x, h = conv(x)
            skips += h

        skips *= math.sqrt(1.0 / len(self.conv_layers))
        x = skips
        x = self.last_conv_layers(x)


        return x


def generator_loss(disc_outputs):
    loss = 0
    gen_losses = []
    for dg in disc_outputs:
        l = torch.mean((1-dg)**2)
        gen_losses.append(l)
        loss += l

    return loss, gen_losses



if __name__ == '__main__':

    model = Generator(1)
    x = torch.ones([2, 1, 16000])
    y = model(x)
    print("Shape of y", y.shape)
    assert x.shape[-1] == y.shape[-1]
