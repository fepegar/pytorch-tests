# Adapted from https://github.com/jvanvugt/pytorch-unet

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F


class UNet(nn.Module):
    def __init__(self, in_channels=1, n_classes=2, depth=5, wf=6, padding=False,
                 batch_norm=False, up_mode='upconv', dimensions=None):
        """
        Implementation of
        U-Net: Convolutional Networks for Biomedical Image Segmentation
        (Ronneberger et al., 2015)
        https://arxiv.org/abs/1505.04597

        Using the default arguments will yield the exact version used
        in the original paper

        Args:
            in_channels (int): number of input channels
            n_classes (int): number of output channels
            depth (int): depth of the network
            wf (int): number of filters in the first layer is 2**wf
            padding (bool): if True, apply padding such that the input shape
                            is the same as the output.
                            This may introduce artifacts
            batch_norm (bool): Use BatchNorm after layers with an
                               activation function
            up_mode (str): one of 'upconv' or 'upsample'.
                           'upconv' will use transposed convolutions for
                           learned upsampling.
                           'upsample' will use bilinear upsampling.
        """
        if dimensions not in (2, 3):
            raise ValueError('dimensions must be 2 or 3, not {}'.format(dimensions))
        super(UNet, self).__init__()
        assert up_mode in ('upconv', 'upsample')
        self.dimensions = dimensions
        self.padding = padding
        self.depth = depth
        prev_channels = in_channels
        self.down_path = nn.ModuleList()
        for i in range(depth):
            self.down_path.append(UNetConvBlock(prev_channels, 2**(wf+i),
                                                padding, batch_norm, dimensions))
            prev_channels = 2**(wf+i)

        self.up_path = nn.ModuleList()
        for i in reversed(range(depth - 1)):
            self.up_path.append(UNetUpBlock(prev_channels, 2**(wf+i), up_mode,
                                            padding, batch_norm, dimensions))
            prev_channels = 2**(wf+i)

        conv_class = nn.Conv2d if dimensions == 2 else nn.Conv3d
        self.last = conv_class(prev_channels, n_classes, kernel_size=1)

    def forward(self, x):
        blocks = []
        for i, down in enumerate(self.down_path):
            x = down(x)
            if i != len(self.down_path)-1:
                blocks.append(x)
                if self.dimensions == 2:
                    x = F.avg_pool2d(x, 2)
                elif self.dimensions == 3:
                    x = F.avg_pool3d(x, 2)

        for i, up in enumerate(self.up_path):
            x = up(x, blocks[-i-1])

        classification = self.last(x)
        probabilities = F.softmax(classification, dim=1)
        return probabilities


    @property
    def num_parameters(self):
        N = sum(np.prod(parameters.shape) for parameters in self.parameters())
        return N


    @property
    def receptive_field(self):
        if self.padding:
            print('Receptive field cannot be computed if padding is enabled')
            return None
        x = self.get_dummy_input()
        y = self(x)
        input_shape = np.array(x.shape[2:])
        output_shape = np.array(y.shape[2:])
        return input_shape - output_shape


class UNetConvBlock(nn.Module):
    def __init__(self, in_size, out_size, padding, batch_norm, dimensions):
        super(UNetConvBlock, self).__init__()
        conv_class = nn.Conv2d if dimensions == 2 else nn.Conv3d
        batch_norm_class = nn.BatchNorm2d if dimensions == 2 else nn.BatchNorm3d

        block = []
        block.append(conv_class(in_size, out_size, kernel_size=3,
                               padding=int(padding)))
        block.append(nn.ReLU())
        if batch_norm:
            block.append(batch_norm_class(out_size))

        block.append(conv_class(out_size, out_size, kernel_size=3,
                               padding=int(padding)))
        block.append(nn.ReLU())
        if batch_norm:
            block.append(batch_norm_class(out_size))

        self.block = nn.Sequential(*block)

    def forward(self, x):
        out = self.block(x)
        return out


class UNetUpBlock(nn.Module):
    def __init__(self, in_size, out_size, up_mode, padding, batch_norm, dimensions):
        super(UNetUpBlock, self).__init__()
        if up_mode == 'upconv':
            conv_class = nn.ConvTranspose2d if dimensions == 2 else nn.ConvTranspose3d
            self.up = conv_class(in_size, out_size, kernel_size=2,
                                         stride=2)
        elif up_mode == 'upsample':
            conv_class = nn.Conv2d if dimensions == 2 else nn.Conv3d
            self.up = nn.Sequential(nn.Upsample(mode='bilinear', scale_factor=2),
                                    nn.Conv2d(in_size, out_size, kernel_size=1))

        self.conv_block = UNetConvBlock(in_size, out_size, padding, batch_norm, dimensions)

    def center_crop(self, layer, target_size):
        dimensions = len(target_size)
        if dimensions == 2:
            _, _, layer_height, layer_width = layer.size()
            diff_y = (layer_height - target_size[0]) // 2
            diff_x = (layer_width - target_size[1]) // 2
            return layer[:,:, diff_y:(diff_y + target_size[0]), diff_x:(diff_x + target_size[1])]
        elif dimensions == 3:
            _, _, layer_depth, layer_height, layer_width = layer.size()
            diff_z = (layer_depth - target_size[0]) // 2
            diff_y = (layer_height - target_size[1]) // 2
            diff_x = (layer_width - target_size[2]) // 2
            return layer[:, :, diff_z:(diff_z + target_size[0]), diff_y:(diff_y + target_size[1]), diff_x:(diff_x + target_size[2])]

    def forward(self, x, bridge):
        up = self.up(x)
        crop1 = self.center_crop(bridge, up.shape[2:])
        out = torch.cat([up, crop1], 1)
        out = self.conv_block(out)

        return out


class UNet2D(UNet):
    def __init__(self, *args, **kwargs):
        kwargs['dimensions'] = 2
        super().__init__(*args, **kwargs)

    def get_dummy_input(self):
        return torch.rand(1, 1, 200, 200)


class UNet3D(UNet):
    def __init__(self, *args, **kwargs):
        kwargs['dimensions'] = 3
        kwargs['depth'] = 4
        kwargs['wf'] = 5
        super().__init__(*args, **kwargs)

    def get_dummy_input(self):
        return torch.rand(1, 1, 100, 100, 100)
