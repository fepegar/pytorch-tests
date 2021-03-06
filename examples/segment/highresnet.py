import torch
from torch import nn
import torch.nn.functional as F
import numpy as np

PADDING_MODES = {
    'reflect': 'Reflection',
    'replicate': 'Replication',
    'constant': 'Zero',
}


class Pad3d(nn.Module):
    def __init__(self, pad, mode):
        assert mode in PADDING_MODES.keys()
        super().__init__()
        self.pad = 6 * [pad]
        self.mode = mode

    def forward(self, x):
        out = F.pad(x, self.pad, self.mode)
        return out



class ConvolutionalBlock(nn.Module):
    def __init__(self, in_channels, out_channels, dilation,
                 dimensions, batch_norm=True, padding_mode='constant'):
        assert padding_mode in PADDING_MODES.keys()
        super().__init__()

        if dimensions == 2:
            padding_class = getattr(nn, f'{PADDING_MODES[padding_mode]}Pad2d')
            padding_instance = padding_class(dilation)
        elif dimensions == 3:
            padding_instance = Pad3d(dilation, padding_mode)
        conv_class = nn.Conv2d if dimensions == 2 else nn.Conv3d
        batch_norm_class = nn.BatchNorm2d if dimensions == 2 else nn.BatchNorm3d

        layers = []
        if batch_norm:
            layers.append(batch_norm_class(in_channels))
        layers.append(nn.ReLU())
        layers.append(padding_instance)
        conv_layer = conv_class(
            in_channels,
            out_channels,
            kernel_size=3,
            dilation=dilation,
        )
        layers.append(conv_layer)
        self.convolutional_block = nn.Sequential(*layers)

    def forward(self, x):
        out = self.convolutional_block(x)
        return out



class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, num_layers, dilation, dimensions,
                 batch_norm=True, residual=True, residual_type='pad'):
        assert residual_type in ('pad', 'project')
        super().__init__()
        self.residual = residual
        self.change_dimension = in_channels != out_channels
        self.residual_type = residual_type
        self.dimensions = dimensions
        if self.change_dimension:
            if residual_type == 'project':
                conv_class = nn.Conv2d if dimensions == 2 else nn.Conv3d
                self.change_dim_layer = conv_class(
                    in_channels,
                    out_channels,
                    kernel_size=1,
                    dilation=dilation,
                )

        conv_blocks = []
        for _ in range(num_layers):
            conv_block = ConvolutionalBlock(
                in_channels,
                out_channels,
                dilation,
                dimensions,
                batch_norm=batch_norm,
            )
            conv_blocks.append(conv_block)
            in_channels = out_channels
        self.residual_block = nn.Sequential(*conv_blocks)

    def forward(self, x):
        """
        From the original ResNet paper, page 4:

        "When the dimensions increase, we consider two options:
        (A) The shortcut stillvperforms identity mapping,
        with extra zero entries padded
        for increasing dimensions. This option introduces no extra
        parameter; (B) The projection shortcut in Eqn.(2) is used to
        match dimensions (done by 1×1 convolutions). For both
        options, when the shortcuts go across feature maps of two
        sizes, they are performed with a stride of 2."
        """
        out = self.residual_block(x)
        if self.residual:
            if self.change_dimension:
                if self.residual_type == 'project':
                    x = self.change_dim_layer(x)
                elif self.residual_type == 'pad':
                    if self.dimensions == 2:
                        N, _, H, W = x.shape
                        pad = out.shape[1] - x.shape[1]  # diff of channels
                        zeros = x.new_zeros(N, pad, H, W)
                    elif self.dimensions == 3:
                        N, _, D, H, W = x.shape
                        pad = out.shape[1] - x.shape[1]  # diff of channels
                        zeros = x.new_zeros(N, pad, D, H, W)
                    x = torch.cat((x, zeros), dim=1)  # channels dimension
            out = x + out
        return out



class DilationBlock(nn.Module):
    def __init__(self, in_channels, out_channels, dilation, dimensions,
                 layers_per_block=2, num_residual_blocks=3, batch_norm=True,
                 residual=True, residual_type='pad'):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        residual_blocks = []
        for _ in range(num_residual_blocks):
            residual_block = ResidualBlock(
                in_channels,
                out_channels,
                layers_per_block,
                dilation,
                dimensions,
                batch_norm=batch_norm,
                residual=residual,
                residual_type=residual_type,
            )
            residual_blocks.append(residual_block)
            in_channels = out_channels
        self.dilation_block = nn.Sequential(*residual_blocks)

    def forward(self, x):
        out = self.dilation_block(x)
        return out



class HighResNet(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 dimensions=None,
                 initial_out_channels_power=4,
                 layers_per_residual_block=2,
                 residual_blocks_per_dilation=3,
                 dilations=3,
                 batch_norm=True,
                 residual=True,
                 residual_type='pad',
                 padding_mode='constant'):
        assert padding_mode in PADDING_MODES.keys()
        assert dimensions in (2, 3)
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.layers_per_residual_block = layers_per_residual_block
        self.residual_blocks_per_dilation = residual_blocks_per_dilation
        self.dilations = dilations

        if dimensions == 2:
            padding_class = getattr(nn, f'{PADDING_MODES[padding_mode]}Pad2d')
            padding_instance = padding_class(1)
        elif dimensions == 3:
            padding_instance = Pad3d(1, padding_mode)
        conv_class = nn.Conv2d if dimensions == 2 else nn.Conv3d
        batch_norm_class = nn.BatchNorm2d if dimensions == 2 else nn.BatchNorm3d

        first_block_list = []
        initial_out_channels = 2 ** initial_out_channels_power
        first_block_list.append(padding_instance)
        first_conv = conv_class(
            in_channels=self.in_channels,
            out_channels=initial_out_channels,
            kernel_size=3,
        )
        first_block_list.append(first_conv)
        if batch_norm:
            first_block_list.append(batch_norm_class(first_conv.out_channels))
        first_block_list.append(nn.ReLU())
        first_block = nn.Sequential(*first_block_list)

        blocks = []
        blocks.append(first_block)
        in_channels = out_channels = first_conv.out_channels
        for dilation_idx in range(dilations):
            if dilation_idx >= 1:
                in_channels = dilation_block.out_channels
            dilation = 2 ** dilation_idx
            dilation_block = DilationBlock(
                in_channels,
                out_channels,
                dilation,
                dimensions,
                layers_per_block=layers_per_residual_block,
                num_residual_blocks=residual_blocks_per_dilation,
                batch_norm=batch_norm,
                residual=residual,
                residual_type=residual_type,
            )
            blocks.append(dilation_block)
            out_channels *= 2

        classifier = conv_class(
            in_channels=blocks[-1].out_channels,
            out_channels=self.out_channels,
            kernel_size=1,
        )
        blocks.append(classifier)
        blocks.append(nn.Softmax(dim=1))
        self.block = nn.Sequential(*blocks)

    def forward(self, x):
        out = self.block(x)
        return out

    @property
    def num_parameters(self):
        return sum(np.prod(p.shape) for p in self.parameters())

    @property
    def receptive_field(self):
        """
        B: number of convolutional layers per residual block
        N: number of residual blocks per dilation factor
        D: number of different dilation factors
        """
        B = self.layers_per_residual_block
        D = self.dilations
        N = self.residual_blocks_per_dilation
        d = np.arange(D)
        input_output_diff = (3 - 1) + np.sum(B * N * 2 ** (d + 1))
        receptive_field = input_output_diff + 1
        return receptive_field

    def get_receptive_field_world(self, spacing=1):
        return self.receptive_field * spacing



class HighRes2DNet(HighResNet):
    def __init__(self, *args, **kwargs):
        kwargs['dimensions'] = 2
        super().__init__(*args, **kwargs)



class HighRes3DNet(HighResNet):
    def __init__(self, *args, **kwargs):
        kwargs['dimensions'] = 3
        super().__init__(*args, **kwargs)



if __name__ == '__main__':
    import torch

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # b = HighRes2DNet(1, 2)
    # b.to(device)
    # b.eval()
    # print(b.num_parameters)
    # i = torch.rand(1, 1, 32, 32, device=device)
    # print(b(i).shape)

    b = HighRes3DNet(
        1, 2,
        initial_out_channels_power=4,
        layers_per_residual_block=2,
        residual_blocks_per_dilation=3,
        dilations=3,
        # residual_type='project',
    )
    b.to(device)
    # b.eval()
    print(b.num_parameters)
    print(b.get_receptive_field_world(spacing=2))
    i = torch.rand(1, 1, 97, 115, 97, device=device)  # 2 mm
    # print(b.get_receptive_field_world(spacing=3))
    # i = torch.rand(1, 1, 64, 76, 64, device=device)  # 3 mm
    # i = torch.rand(1, 1, 80, 80, 80, device=device)
    print(b(i).shape)
