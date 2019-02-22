import torch
import torch.nn as nn
import torch.nn.functional as F


class LeNet(nn.Module):

    def __init__(self):
        super(LeNet, self).__init__()

        self.output_cnn_size = 5

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
            in_features=self.output_cnn_size**2 * self.conv2.out_channels,
            out_features=120,
        )
        self.fc2 = nn.Linear(
            in_features=120,
            out_features=84,
        )
        self.fc3 = nn.Linear(
            in_features=84,
            out_features=10,
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


if __name__ == '__main__':
    net = LeNet()
    print(net)
    nSamples = 1
    nChannels = 1
    height = 32
    width = 32
    net_input = torch.randn(
        nSamples,
        nChannels,
        height,
        width,
    )
    net_output = net(net_input)
    print(net_output)

    # TODO: understand these two lines
    net.zero_grad()
    net_output.backward(torch.randn(1, 10))
