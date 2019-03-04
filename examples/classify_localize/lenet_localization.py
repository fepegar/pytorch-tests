import torch.nn as nn
import torch.nn.functional as F

import numpy as np


class LeNetLocalization(nn.Module):

    def __init__(self, initial_shape, output_units=4):
        super(LeNetLocalization, self).__init__()
        self.initial_shape = np.array(initial_shape)
        self.output_cnn_size = self.final_shape(self.initial_shape).prod()

        self.output_units = output_units

        # 1 input image channel, 6 output channels, 5x5 square convolution
        # kernel
        self.conv1 = nn.Conv2d(
            in_channels=1,
            out_channels=6,
            kernel_size=5,
        )
        self.conv2 = nn.Conv2d(
            in_channels=self.conv1.out_channels,
            out_channels=16,
            kernel_size=5,
        )
        self.fc1 = nn.Linear(
            in_features=self.output_cnn_size * self.conv2.out_channels,
            out_features=120,
        )
        self.fc2 = nn.Linear(
            in_features=self.fc1.out_features,
            out_features=84,
        )
        self.fc3 = nn.Linear(
            in_features=self.fc2.out_features,
            out_features=self.output_units,
        )

    def forward(self, x):
        x = F.max_pool2d(
            F.relu(self.conv1(x)),
            (2, 2),
        )
        x = F.max_pool2d(
            F.relu(self.conv2(x)),
            (2, 2),
        )
        new_shape = -1, self.num_flat_features(x)
        x = x.view(new_shape)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    @staticmethod
    def num_flat_features(x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

    @staticmethod
    def final_shape(initial_shape):
        shape = initial_shape.copy()
        for i in range(2):
            shape -= 4  # conv
            shape //= 2  # pool
        return shape

    @property
    def num_parameters(self):
        return sum(parameters.shape for parameters in self.parameters())
