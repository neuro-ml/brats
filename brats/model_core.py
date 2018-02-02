import torch.nn as nn

from dpipe.model_core.layers_torch.blocks import ConvBlock3d

conv_block = ConvBlock3d


class SimpleNet(nn.Module):
    def __init__(self, n_chans_in, n_chans_out, activation, structure):
        super().__init__()

        structure = [n_chans_in, *structure]
        kernel_size = 3

        path = [conv_block(n_chans_in=a, n_chans_out=b, activation=activation, kernel_size=kernel_size)
                for a, b in zip(structure[:-1], structure[1:])]

        self.path = nn.Sequential(*path,
                                  conv_block(n_chans_in=structure[-1], n_chans_out=n_chans_out, kernel_size=1))

    def forward(self, x):
        return self.path(x)
