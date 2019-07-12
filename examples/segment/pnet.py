"""
Interactive Medical Image Segmentation Using Deep Learning With Image-Specific Fine Tuning
DeepIGeoS: A Deep Interactive Geodesic Framework for Medical Image Segmentation
"""

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F


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
                 dimensions, padding_mode='constant'):
        assert padding_mode in PADDING_MODES.keys()
        super().__init__()

        if dimensions == 2:
            padding_class = getattr(nn, f'{PADDING_MODES[padding_mode]}Pad2d')
            padding_layer = padding_class(dilation)
        elif dimensions == 3:
            padding_layer = Pad3d(dilation, padding_mode)
        conv_class = nn.Conv2d if dimensions == 2 else nn.Conv3d
        conv_layer = conv_class(
            in_channels,
            out_channels,
            kernel_size=3,
            dilation=dilation,
        )
        self.convolutional_block = nn.Sequential(
            padding_layer,
            conv_layer,
            nn.ReLU(),
        )

    def forward(self, x):
        out = self.convolutional_block(x)
        return out



class PNetBlock(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 dilation,
                 dimensions,
                 num_layers):
        super().__init__()
        blocks = nn.ModuleList()
        for _ in range(num_layers):
            conv_block = ConvolutionalBlock(
                in_channels,
                out_channels,
                dilation,
                dimensions,
            )
            blocks.append(conv_block)
            in_channels = out_channels
        self.block = nn.Sequential(*blocks)
        if dimensions == 3:
            self.reduce_channels_conv = nn.Conv3d(
                in_channels=out_channels,
                out_channels=out_channels // 4,
                kernel_size=1,
            )

    def forward(self, x):
        out = self.block(x)
        return out



class BasePNet(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 default_num_channels=64,
                 dimensions=None):
        assert dimensions in (2, 3)
        super().__init__()
        self.out_channels = out_channels
        self.dimensions = dimensions
        layers_per_block = 2, 2, 3, 3, 3
        self.blocks = nn.ModuleList()
        out_channels = default_num_channels
        for i, num_layers in enumerate(layers_per_block):
            dilation = 2 ** i if dimensions == 2 else i + 1
            pnet_block = PNetBlock(
                in_channels,
                out_channels,
                dilation,
                dimensions,
                num_layers,
            )
            self.blocks.append(pnet_block)
            in_channels = out_channels

        output_blocks = nn.ModuleList()
        if dimensions == 2:
            in_channels = out_channels * len(self.blocks)
            out_channels *= 2
            output_blocks.append(nn.Dropout2d())
            conv_1 = nn.Conv2d(in_channels, out_channels, kernel_size=1)
            output_blocks.append(conv_1)
            output_blocks.append(nn.Dropout2d())
            conv_2 = nn.Conv2d(conv_1.out_channels,
                               self.out_channels, kernel_size=1)
            output_blocks.append(conv_2)
            output_blocks.append(nn.Softmax2d())
        elif dimensions == 3:
            in_channels = (out_channels // 4) * len(self.blocks)
            output_blocks.append(nn.Dropout3d())
            conv = nn.Conv3d(in_channels, self.out_channels, kernel_size=1)
            output_blocks.append(conv)
            output_blocks.append(nn.Softmax(dim=1))  # channels
        self.output_block = nn.Sequential(*output_blocks)


    def forward(self, x):
        outputs = []
        for block in self.blocks:
            x = block(x)
            if self.dimensions == 2:
                outputs.append(x)
            elif self.dimensions == 3:
                block_output = block.reduce_channels_conv(x)
                outputs.append(block_output)
        x = torch.cat(outputs, dim=1)
        out = self.output_block(x)
        return out

    @property
    def num_parameters(self):
        N = sum(np.prod(parameters.shape) for parameters in self.parameters())
        return N



class PNet(BasePNet):
    def __init__(self, *args, **kwargs):
        kwargs['dimensions'] = 2
        super().__init__(*args, **kwargs)



class PCNet(BasePNet):
    def __init__(self, *args, **kwargs):
        kwargs['dimensions'] = 3
        super().__init__(*args, **kwargs)


if __name__ == '__main__':
    import torch

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # b = PNet(1, 2)
    # b.to(device)
    # b.eval()
    # print(b.num_parameters)
    # i = torch.rand(1, 1, 32, 32, device=device)
    # print(b(i).shape)

    b = PCNet(
        1, 2,
    )
    b.to(device)
    # b.eval()
    print(b.num_parameters)
    i = torch.rand(1, 1, 97, 115, 97, device=device)  # 2 mm
    # print(b.get_receptive_field_world(spacing=3))
    # i = torch.rand(1, 1, 64, 76, 64, device=device)  # 3 mm
    # i = torch.rand(1, 1, 80, 80, 80, device=device)
    print(b(i).shape)
