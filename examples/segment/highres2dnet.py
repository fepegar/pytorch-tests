import torch
from torch import nn

padding_modes = (
    'Reflection',
    'Replication',
    'Zero',
)


class HighRes2DNet(nn.Module):
    def __init__(self,
                 in_channels=1,
                 out_channels=2,
                 initial_out_channels_power=4,
                 layers_per_block=2,
                 blocks_per_dilation=3,
                 dilations=3,
                 batch_norm=True,
                 residual=True,
                 padding_mode='Zero'):
        assert padding_mode in padding_modes
        super(HighRes2DNet, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        first_block_list = nn.ModuleList()  # necessary?
        initial_out_channels = 2 ** initial_out_channels_power
        padding_class = getattr(nn, f'{padding_mode}Pad2d')
        first_block_list.append(padding_class(1))
        first_conv = nn.Conv2d(
            in_channels=self.in_channels,
            out_channels=initial_out_channels,
            kernel_size=3,
        )
        first_block_list.append(first_conv)
        if batch_norm:
            first_block_list.append(nn.BatchNorm2d(first_conv.out_channels))
        first_block_list.append(nn.ReLU())
        first_block = nn.Sequential(*first_block_list)

        blocks = nn.ModuleList()  # necessary?
        blocks.append(first_block)
        in_channels = out_channels = first_conv.out_channels
        for dilation_idx in range(dilations):
            if dilation_idx >= 1:
                in_channels = dilation_block.out_channels
            dilation = 2 ** dilation_idx
            dilation_block = DilationBlock(
                in_channels=in_channels,
                out_channels=out_channels,
                dilation=dilation,
                layers_per_block=layers_per_block,
                num_residual_blocks=blocks_per_dilation,
                batch_norm=batch_norm,
                residual=residual,
            )
            blocks.append(dilation_block)
            out_channels *= 2

        classifier = nn.Conv2d(
            in_channels=blocks[-1].out_channels,
            out_channels=self.out_channels,
            kernel_size=1,
        )
        blocks.append(classifier)
        blocks.append(nn.Softmax2d())
        self.block = nn.Sequential(*blocks)

    def forward(self, x):
        out = self.block(x)
        return out

    @property
    def num_parameters(self):
        import numpy as np
        return sum(np.prod(p.shape) for p in self.parameters())


class DilationBlock(nn.Module):
    def __init__(self, in_channels, out_channels, dilation,
                 layers_per_block=2, num_residual_blocks=3, batch_norm=True,
                 residual=True):
        super(DilationBlock, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        residual_blocks = nn.ModuleList()  # necessary?
        for _ in range(num_residual_blocks):
            residual_block = ResidualBlock(
                in_channels=in_channels,
                out_channels=out_channels,
                num_layers=layers_per_block,
                dilation=dilation,
                batch_norm=batch_norm,
                residual=residual,
            )
            residual_blocks.append(residual_block)
            in_channels = out_channels
        self.dilation_block = nn.Sequential(*residual_blocks)

    def forward(self, x):
        out = self.dilation_block(x)
        return out


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, num_layers, dilation,
                 batch_norm=True, residual=True, residual_type='pad'):
        assert residual_type in ('pad', 'project')
        super(ResidualBlock, self).__init__()
        self.residual = residual
        self.change_dimension = in_channels != out_channels
        self.residual_type = residual_type
        if self.change_dimension:
            if residual_type == 'project':
                self.change_dim_layer = nn.Conv2d(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=1,
                    dilation=dilation,
                )

        conv_blocks = nn.ModuleList()  # necessary?
        for _ in range(num_layers):
            conv_block = ConvolutionalBlock(
                in_channels=in_channels,
                out_channels=out_channels,
                dilation=dilation,
                batch_norm=batch_norm,
            )
            conv_blocks.append(conv_block)
            in_channels = out_channels
        self.residual_block = nn.Sequential(*conv_blocks)

    def forward(self, x):
        out = self.residual_block(x)
        if self.residual:
            if self.change_dimension:
                if self.residual_type == 'project':
                    x = self.change_dim_layer(x)
                elif self.residual_type == 'pad':
                    N, _, H, W = x.shape
                    pad = out.shape[1] - x.shape[1]  # diff of channels
                    zeros = torch.zeros(N, pad, H, W)
                    x = torch.cat((x, zeros), dim=1)  # channels dimension
            out = x + out
        return out


class ConvolutionalBlock(nn.Module):
    def __init__(self, in_channels, out_channels, dilation,
                 batch_norm=True, padding_mode='Zero'):
        assert padding_mode in padding_modes
        super(ConvolutionalBlock, self).__init__()
        layers = nn.ModuleList()  # necessary?
        if batch_norm:
            layers.append(nn.BatchNorm2d(in_channels))
        layers.append(nn.ReLU())
        padding_class = getattr(nn, f'{padding_mode}Pad2d')
        layers.append(padding_class(dilation))
        conv_layer = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=3,
            dilation=dilation,
        )
        layers.append(conv_layer)
        self.convolutional_block = nn.Sequential(*layers)

    def forward(self, x):
        out = self.convolutional_block(x)
        return out


if __name__ == '__main__':
    import torch
    b = HighRes2DNet()
    print(b)
    i = torch.rand(1, 1, 256, 256)
    print(b(i).shape)
