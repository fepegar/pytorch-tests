import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

UNET2D = '2D'
UNET3D = '3D'

class UNet(nn.Module):
    def __init__(self, num_levels=3, unet_features_type=UNET2D):
        super(UNet, self).__init__()
        self.num_levels = num_levels

        ### Define layers ###
        ## Contracting path ##
        # Level 1 down
        out_channels = 32 if unet_features_type == UNET3D else 64
        self.conv_down_1_0 = nn.Conv2d(
            in_channels=1,
            out_channels=out_channels,
            kernel_size=3,
            padding=1,
        )
        if unet_features_type == UNET3D:
            out_channels *= 2
        self.conv_down_1_1 = nn.Conv2d(
            in_channels=self.conv_down_1_0.out_channels,
            out_channels=out_channels,
            kernel_size=3,
            padding=1,
        )

        # Level 2 down
        if unet_features_type == UNET2D:
            out_channels *= 2
        self.conv_down_2_0 = nn.Conv2d(
            in_channels=self.conv_down_1_1.out_channels,
            out_channels=out_channels,
            kernel_size=3,
            padding=1,
        )
        if unet_features_type == UNET3D:
            out_channels *= 2
        self.conv_down_2_1 = nn.Conv2d(
            in_channels=self.conv_down_2_0.out_channels,
            out_channels=out_channels,
            kernel_size=3,
            padding=1,
        )

        # Level 3 down
        if unet_features_type == UNET2D:
            out_channels *= 2
        self.conv_down_3_0 = nn.Conv2d(
            in_channels=self.conv_down_2_1.out_channels,
            out_channels=out_channels,
            kernel_size=3,
            padding=1,
        )
        if unet_features_type == UNET3D:
            out_channels *= 2
        self.conv_down_3_1 = nn.Conv2d(
            in_channels=self.conv_down_3_0.out_channels,
            out_channels=out_channels,
            kernel_size=3,
            padding=1,
        )

        ## Expanding path ##
        # Level 3 up
        if unet_features_type == UNET2D:
            out_channels *= 2
        self.conv_up_3_0 = nn.Conv2d(
            in_channels=self.conv_down_3_1.out_channels,
            out_channels=out_channels,
            kernel_size=3,
            padding=1,
        )

        if unet_features_type == UNET3D:
            out_channels *= 2
        self.conv_up_3_1 = nn.Conv2d(
            in_channels=self.conv_up_3_0.out_channels,
            out_channels=out_channels,
            kernel_size=3,
            padding=1,
        )

        self.deconv_3 = nn.ConvTranspose2d(
            in_channels=self.conv_up_3_1.out_channels,
            out_channels=self.conv_up_3_1.out_channels,
            kernel_size=2,
            stride=2,
        )

        # Level 2 up
        in_channels = (
            self.conv_down_3_1.out_channels
            + self.conv_up_3_1.out_channels
        )
        out_channels //= 2
        self.conv_up_2_0 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=3,
            padding=1,
        )
        self.conv_up_2_1 = nn.Conv2d(
            in_channels=self.conv_up_2_0.out_channels,
            out_channels=out_channels,
            kernel_size=3,
            padding=1,
        )
        self.deconv_2 = nn.ConvTranspose2d(
            in_channels=self.conv_up_2_1.out_channels,
            out_channels=self.conv_up_2_1.out_channels,
            kernel_size=2,
            stride=2,
        )

        # Level 1 up
        in_channels = (
            self.conv_down_2_1.out_channels
            + self.conv_up_2_1.out_channels
        )
        out_channels //= 2
        self.conv_up_1_0 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=3,
            padding=1,
        )
        self.conv_up_1_1 = nn.Conv2d(
            in_channels=self.conv_up_1_0.out_channels,
            out_channels=out_channels,
            kernel_size=3,
            padding=1,
        )
        self.deconv_1 = nn.ConvTranspose2d(
            in_channels=self.conv_up_1_1.out_channels,
            out_channels=self.conv_up_1_1.out_channels,
            kernel_size=2,
            stride=2,
        )

        # Last conv layers
        in_channels = (
            self.conv_down_1_1.out_channels
            + self.conv_up_1_1.out_channels
        )
        out_channels //= 2
        self.conv_out_1 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=3,
            padding=1,
        )
        self.conv_out_2 = nn.Conv2d(
            in_channels=self.conv_out_1.out_channels,
            out_channels=out_channels,
            kernel_size=3,
            padding=1,
        )
        self.conv_out_3 = nn.Conv2d(
            in_channels=self.conv_out_2.out_channels,
            out_channels=2,
            kernel_size=1,
        )

    def forward(self, x):
        # self.assert_input_shape(x)

        # Level 1
        x = F.relu(self.conv_down_1_0(x))
        x = out_1 = F.relu(self.conv_down_1_1(x))
        x = F.max_pool2d(x, 2)

        # Level 2
        x = F.relu(self.conv_down_2_0(x))
        x = out_2 = F.relu(self.conv_down_2_1(x))
        x = F.max_pool2d(x, 2)

        # Level 3
        x = F.relu(self.conv_down_3_0(x))
        x = out_3 = F.relu(self.conv_down_3_1(x))
        x = F.max_pool2d(x, 2)

        # Level 3
        x = F.relu(self.conv_up_3_0(x))
        x = F.relu(self.conv_up_3_1(x))
        x = self.deconv_3(x, output_size=out_3.size())
        x = torch.cat((out_3, x), dim=1)  # dim 1 is channels

        # Level 2
        x = F.relu(self.conv_up_2_0(x))
        x = F.relu(self.conv_up_2_1(x))
        x = self.deconv_2(x, output_size=out_2.size())
        x = torch.cat((out_2, x), dim=1)  # dim 1 is channels

        # Level 1
        x = F.relu(self.conv_up_1_0(x))
        x = F.relu(self.conv_up_1_1(x))
        x = self.deconv_1(x, output_size=out_1.size())
        x = torch.cat((out_1, x), dim=1)  # dim 1 is channels

        # Output
        x = F.relu(self.conv_out_1(x))
        x = F.relu(self.conv_out_2(x))
        classification = self.conv_out_3(x)

        probabilities = F.softmax(classification, dim=1)

        return probabilities

    def assert_input_shape(self, x):
        _, _, height, width = x.shape
        factor = 2 ** self.num_levels
        height_ok = height % factor == 0
        width_ok = width % factor == 0
        message = (
            f'Input image size must be divisible by {factor}'
            f' but it is ({height}, {width})'
        )
        assert height_ok and width_ok, message

    @property
    def num_parameters(self):
        N = sum(np.prod(parameters.shape) for parameters in self.parameters())
        return N
