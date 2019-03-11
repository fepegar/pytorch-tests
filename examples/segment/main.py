import logging
from pathlib import Path

import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from tensorboardX import SummaryWriter

import torch
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision.utils import make_grid

from unet import UNet
from dataset import ResectionSideDataset


def get_batch(loader):
    return next(iter(loader))


def plot_batch(batch, outputs=None):
    images = batch.image
    labels = batch.label
    images_grid = make_grid(
        images,
        normalize=True,
        scale_each=True,
        padding=0,
    )
    images_grid = images_grid.numpy().transpose(1, 2, 0)
    labels_grid = make_grid(labels, padding=0).numpy().transpose(1, 2, 0)
    labels_grid = labels_grid[..., 0]  # Keep only one channel since grayscale
    images_grid[..., 1][labels_grid > 0] = 1  # green
    if outputs is not None:
        foreground = outputs[:, 1:2, ...]
        foreground = foreground > 0.5  # foreground
        outputs_grid = make_grid(foreground, padding=0).numpy().transpose(1, 2, 0)
        outputs_grid = outputs_grid[..., 0]  # Keep only one channel since grayscale
        images_grid[..., 0][outputs_grid > 0] = 1  # red
        images_grid[..., 2][outputs_grid > 0] = 1  # blue
    fig, axis = plt.subplots()
    axis.imshow(images_grid)
    axis.set_title(' '.join(batch.name))
    plt.tight_layout()
    return fig


class ColoredFormatter(logging.Formatter):
    def __init__(self, msg, use_color=True):
        super().__init__(msg)
        self.use_color = use_color
        RED = '\033[91m'
        GREEN = '\033[92m'
        YELLOW = '\033[93m'
        BLUE = '\033[94m'
        MAGENTA = '\033[95m'
        CYAN = '\033[96m'
        WHITE = '\033[97m'
        BOLD = '\033[1m'
        ENDC = '\033[0m'

        self.colormap = {
            'INFO': GREEN,
            'DEBUG': CYAN,
            'WARNING': YELLOW,
            'ERROR': RED,
            'CRITICAL': RED,
        }

    def format(self, record):
        levelname = record.levelname
        ENDC = '\033[0m'
        if self.use_color:
            try:
                levelname_color = self.colormap[levelname] + levelname + ENDC
                record.levelname = levelname_color
            except KeyError:
                print('levelname not in', self.colormap)
        return logging.Formatter.format(self, record)


if __name__ == '__main__':
    batch_size = 4
    learning_rate = 1e-3

    # pylint: disable=invalid-name
    verbose = False

    # TensorBoard
    writer = SummaryWriter()
    logger.debug(f'TensorBoard directory: {writer.log_dir}')
    experiment_dir = Path(writer.log_dir)

    # Log
    log_path = '/tmp/log.log'  # experiment_dir / 'log.log'
    logger.debug(f'Log file: {log_path}')

    logger = logging.getLogger(experiment_dir.name)
    logger.setLevel(logging.DEBUG)

    log_file = logging.FileHandler(log_path)
    log_file.setLevel(logging.DEBUG)
    formatter = logging.Formatter(
        '%(asctime)s - %(levelname)-8s - %(message)s')
    log_file.setFormatter(formatter)
    logger.addHandler(log_file)

    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    console_formatter = ColoredFormatter('%(levelname)-8s - %(message)s')
    console.setFormatter(console_formatter)
    logger.addHandler(console)

    # Sampling log
    sampled_images_log_path = experiment_dir / 'sampled_images.log'
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
    training_indices = training_indices[:8]
    validation_indices = validation_indices[:4]
    training_set = Subset(dataset, training_indices)
    validation_set = Subset(dataset, validation_indices)

    # Create loaders
    num_workers = 0  # 0 if CUDA?  -- Also > 0 doesn't work with pytorch-nightly
    training_loader = DataLoader(
        training_set,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=True,
    )

    # I can use larger batch for validation as no gradient needs to be computed
    validation_loader = DataLoader(
        validation_set,
        batch_size=batch_size * 2,
        num_workers=num_workers,
        shuffle=False,  # no need to shuffle the validation set
    )

    # UNet
    # net = UNet()
    # from unet_github import UNet
    # net = UNet(
    #     depth=4,
    #     padding=True,
    # )
    from highresnet import HighRes2DNet
    # net = HighRes2DNet(in_channels=1,
    #                    out_channels=2,
    #                    initial_out_channels_power=3,
    #                    layers_per_block=2,
    #                    blocks_per_dilation=2,
    #                    dilations=3,
    #                    batch_norm=True,
    #                    residual=True,
    #                    padding_mode='constant',
    # )
    from pnet import PNet
    net = PNet(in_channels=1, out_channels=2)

    # # Log architecture
    # if True: #verbose:
    #     dummy_input = torch.zeros(1, 1, net_input_size, net_input_size)
    #     writer.add_graph(
    #         net,
    #         input_to_model=dummy_input,
    #         # verbose=True,
    #     )

    # Loss
    from dice import dice_loss
    criterion = dice_loss

    # Optimizer
    optimizer = optim.Adam(
        net.parameters(),
        lr=learning_rate,
        amsgrad=True,
    )

    logger.info('Training...')
    epochs = 100
    epochs += 1  # For logging purposes
    log_every_n_epochs = 10
    progress = tqdm(range(epochs))
    iteration = 0
    try:
        for epoch in progress:  # loop over the dataset multiple times
            # Log epoch validation loss
            net.eval()
            validation_loss = 0
            for batch_index, batch in enumerate(validation_loader):
                inputs = batch.image
                labels = batch.label
                outputs = net(inputs)
                foreground = outputs[:, 1:2, ...]
                batch_loss = criterion(foreground, labels.float())
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
                labels = batch.label
                sampled_images.append(batch.name)

                # Zero the parameter gradients
                optimizer.zero_grad()

                # Forward pass
                outputs = net(inputs)

                # Compute loss
                foreground = outputs[:, 1:2, ...]
                batch_loss = criterion(foreground, labels.float())

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

            sampled_images_log_path.write_text('\n'.join(lines))
        logger.info('Finished training')
    except KeyboardInterrupt:
        logger.debug('Test')
        logger.warning('KeyboardInterrupt')
    except Exception as e:
        logger.critical('Exception occurred', exc_info=True)

    model_path = experiment_dir / 'model.pt'
    logger.info(f'Saving model to "{model_path}"...')
    torch.save(net.state_dict(), model_path)
