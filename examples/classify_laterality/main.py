from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.utils import make_grid
from torchvision.transforms import Pad, Resize, Compose

from lenet import LeNet
from dataset import ResectionSideDataset

if __name__ == '__main__':

    root_dir = Path('~/git/pytorch-tests/pytorch_dataset/').expanduser()
    csv_path = root_dir / 'data.csv'
    native_shape = np.array((193, 229))
    difference = np.array((256, 256)) - native_shape
    ini = ini_i, ini_j = difference // 2
    fin_i, fin_j = difference - ini
    transform = Compose([
        Pad((ini_i, ini_j, fin_i, fin_j), padding_mode='edge'),
        Resize((32, 32), Image.LANCZOS),
    ])
    dataset = ResectionSideDataset(str(root_dir), transform=transform)

    loader = DataLoader(
        dataset,
        batch_size=4,
        shuffle=True,
        # num_workers=4,
    )

    def get_batch(loader):
        return next(iter(loader))

    def plot_batch(batch, classes=None):
        images = batch['image']
        grid = make_grid(images)
        image = grid.numpy().transpose((1, 2, 0))
        image -= image.min()
        image /= image.max()
        if classes is not None:
            classes = np.array(classes)[batch['target']]
            plt.title(' '.join(classes))
        plt.imshow(image)


    batch = get_batch(loader)
    _, _, height, width = batch['image'].shape
    shape = height, width
    shape = 32, 32
    net = LeNet(shape)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

    def predict(inputs, network, classes):
        # inputs shape is [batch_size, channels, height, width]
        if len(inputs.shape) == 3:  # [channels, height, width]
            inputs = torch.unsqueeze(inputs, 0)
        outputs = network(inputs)
        _, predicted = outputs.max(dim=1)
        return np.array(classes)[predicted]


    print('Training...')
    progress = tqdm(range(40))
    for epoch in progress:  # loop over the dataset multiple times
        running_loss = 0
        for i, batch in enumerate(loader):
            inputs = batch['image']
            labels = dataset.classes[batch['target']]

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if i % 10 == 9:  # print every 10 mini-batches
                message = '[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss / 10)
                progress.set_description(message)
                running_loss = 0

    print('Finished Training')
