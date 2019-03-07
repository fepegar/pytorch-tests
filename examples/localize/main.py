from pathlib import Path

import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from tensorboardX import SummaryWriter

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision.utils import make_grid

from lenet_localization import LeNetLocalization
from dataset import ResectionSideDataset

# TODO:
# Same on 3D


def get_batch(loader):
    return next(iter(loader))


def add_boxes_to_axis(axis, images, boxes, color=(0, 1, 0)):
    N, C, H, W = images.shape
    for column, box in enumerate(boxes):
        center_y, center_x, height, width = box.detach().numpy()
        x_ini = (center_x - width / 2 + column) * W
        y_ini = (center_y - height / 2) * H
        height *= H
        width *= W
        xy = x_ini, y_ini
        rectangle = Rectangle(xy, width, height,
                              edgecolor=color, facecolor='none')
        axis.add_patch(rectangle)


def plot_batch(batch, outputs=None):
    images = batch.image
    labels = batch.label
    boxes = batch.bounding_box
    images_grid = make_grid(
        images,
        normalize=True,
        scale_each=True,
        padding=0,
    )
    images_grid = images_grid.numpy().transpose(1, 2, 0)
    labels_grid = make_grid(labels, padding=0).numpy().transpose(1, 2, 0)
    images_grid[labels_grid > 0] = 1
    fig, axis = plt.subplots()
    axis.imshow(images_grid)
    add_boxes_to_axis(axis, images, boxes)
    if outputs is not None:
        add_boxes_to_axis(axis, images, outputs, color=(1, 0, 1))
    axis.set_title(' '.join(batch.name))
    plt.tight_layout()
    return fig


if __name__ == '__main__':
    # pylint: disable=invalid-name
    verbose = False

    # TensorBoard
    writer = SummaryWriter()

    # Sampling log
    log_path = Path(writer.log_dir) / 'log.txt'
    sampled_images = []

    # Define input image size
    net_input_size = 128

    # Figure out necessary transforms to get a 32x32 image
    native_shape = np.array((193, 229))
    difference = np.array((256, 256)) - native_shape
    ini = top, left = difference // 2
    bottom, right = difference - ini
    padding = (left, top, right, bottom)

    # Create datasets
    root_dir = Path('/tmp/pytorch_dataset/').expanduser()
    dataset = ResectionSideDataset(root_dir, padding, resize=net_input_size)
    N = len(dataset)
    dataset_split_ratio = 0.8
    all_indices = np.arange(N)
    np.random.shuffle(all_indices)
    first_validation_index = int(N * dataset_split_ratio)
    training_indices = all_indices[:first_validation_index]
    validation_indices = all_indices[first_validation_index:]
    training_set = Subset(dataset, training_indices)
    validation_set = Subset(dataset, validation_indices)

    # Create loaders
    num_workers = 0  # 0 if CUDA?  -- Also > 0 doesn't work with pytorch-nightly
    batch_size = 4
    training_loader = DataLoader(
        training_set,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=True,
    )

    # I can use larger batch for validation as no gradient needs to be computed
    validation_loader = DataLoader(
        validation_set,
        batch_size=2 * batch_size,
        num_workers=num_workers,
        shuffle=False,  # no need to shuffle the validation set
    )

    # LeNet
    batch = get_batch(training_loader)
    _, _, height, width = batch.image.shape
    shape = height, width
    shape = net_input_size, net_input_size

    # https://discuss.pytorch.org/t/how-can-l-load-my-best-model-as-a-feature-extractor-evaluator/17254/6
    activation = {}

    def get_activation(name):
        def hook(model, input, output):
            activation[name] = output.detach()
        return hook

    # Actual network
    net = LeNetLocalization(shape)
    net.fc2.register_forward_hook(get_activation('fc2'))
    net.conv2.register_forward_hook(get_activation('conv2'))

    # Log architecture
    if verbose:
        dummy_input = torch.zeros(1, 1, net_input_size, net_input_size)
        writer.add_graph(
            net,
            input_to_model=dummy_input,
            # verbose=True,
        )

    # Loss
    criterion = nn.MSELoss()

    # Optimizer
    optimizer = optim.Adam(
        net.parameters(),
        lr=1e-3,
        amsgrad=True,
    )

    print('Training...')
    epochs = 41
    log_every_n_epochs = 10
    progress = tqdm(range(epochs))
    iteration = 0
    features_shape = len(dataset), net.fc2.out_features
    for epoch in progress:  # loop over the dataset multiple times
        # Log epoch validation loss
        net.eval()
        validation_loss = 0
        for batch_index, batch in enumerate(validation_loader):
            inputs = batch.image
            bounding_boxes = batch.bounding_box
            outputs = net(inputs)
            batch_loss = criterion(outputs, bounding_boxes)
            validation_loss += batch_loss.item()

            # Plot first batch
            if batch_index == 0:
                figure = plot_batch(batch, outputs=outputs)
                writer.add_figure(
                    'Epoch/Validation',
                    figure,
                    iteration,
                )
        num_validation_batches = len(validation_loader)
        validation_loss /= num_validation_batches
        writer.add_scalars(
            'Loss',
            {'Epoch/Validation': validation_loss},
            iteration,
        )

        # Log stuff every N epochs
        if epoch % log_every_n_epochs == 0 and verbose:
            # Log weights histograms
            for name, parameters in net.named_parameters():
                writer.add_histogram(
                    name,
                    parameters.detach().numpy(),
                    iteration,
                )

        # Update weights
        net.train()
        running_loss = 0
        epoch_loss = 0
        for batch_index, batch in enumerate(training_loader):
            iteration += 1
            inputs = batch.image
            bounding_boxes = batch.bounding_box
            sampled_images.append(batch.name)

            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = net(inputs)

            # Compute loss
            batch_loss = criterion(outputs, bounding_boxes)

            # Compute gradients (dJ/dw) through backpropagation
            batch_loss.backward()

            # Update weights
            optimizer.step()

            # Print statistics
            running_loss += batch_loss.item()
            if iteration % 10 == 9:  # print every 10 iterations
                running_loss /= 10
                message = f'[{epoch}, {batch_index:5d}] loss: {running_loss:.3f}'
                progress.set_description(message)
                running_loss = 0

            # Log iteration loss
            writer.add_scalars(
                'Loss',
                {'Batch/Training': batch_loss.item()},
                iteration,
            )
            epoch_loss += batch_loss.item()

            # Plot first batch
            if batch_index == 0:
                figure = plot_batch(batch, outputs=outputs)
                writer.add_figure(
                    'Epoch/Training',
                    figure,
                    iteration,
                )

        # Log epoch training loss
        num_training_batches = len(training_loader)
        epoch_loss /= num_training_batches
        writer.add_scalars(
            'Loss',
            {'Epoch/Training': epoch_loss},
            iteration,
        )

        # Update log
        lines = []
        for iteration, images_names in enumerate(sampled_images):
            lines.append(
                f'Iteration {iteration: 5d}: {", ".join(images_names)}')

        log_path.write_text('\n'.join(lines))

    print()
    print('Finished Training')

    model_path = Path(writer.log_dir) / 'model.pt'
    print(f'Saving model to {model_path}...')
    torch.save(net.state_dict(), model_path)
