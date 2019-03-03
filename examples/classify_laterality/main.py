from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt
from tensorboardX import SummaryWriter

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision.utils import make_grid
from torchvision.transforms import Pad, Resize, Compose

from lenet import LeNet
from dataset import ResectionSideDataset

# TODO:
# Add validation
# Try detection
# Try segmentation
# Same on 3D


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


def predict(inputs, network, classes):
    # inputs shape should be [batch_size, channels, height, width]
    if len(inputs.shape) == 3:  # [channels, height, width]
        inputs = torch.unsqueeze(inputs, 0)
    outputs = network(inputs)
    _, predicted = outputs.max(dim=1)
    return np.array(classes)[predicted]


if __name__ == '__main__':
    # pylint: disable=invalid-name
    verbose = False

    # TensorBoard
    writer = SummaryWriter()

    # Sampling log
    log_path = Path(writer.log_dir) / 'log.txt'
    sampled_images = []

    # Define input image size
    net_input_size = 64

    # Figure out necessary transforms to get a 32x32 image
    native_shape = np.array((193, 229))
    difference = np.array((256, 256)) - native_shape
    ini = top, left = difference // 2
    bottom, right = difference - ini
    padding = (left, top, right, bottom)
    transform = Compose([
        Pad(padding, padding_mode='edge'),
        Resize((net_input_size, net_input_size), Image.LANCZOS),
    ])

    # Create datasets
    root_dir = Path('/tmp/pytorch_dataset/').expanduser()
    dataset = ResectionSideDataset(str(root_dir), transform=transform)
    N = len(dataset)
    dataset_split_ratio = 0.8
    all_indices = np.arange(N)
    np.random.shuffle(all_indices)
    first_validation_index = int(N * dataset_split_ratio)
    training_indices = all_indices[first_validation_index:]
    validation_indices = all_indices[:first_validation_index]
    training_set = Subset(dataset, training_indices)
    validation_set = Subset(dataset, validation_indices)

    # Create loaders
    num_workers = 4  # 0 if CUDA?
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
    _, _, height, width = batch['image'].shape
    shape = height, width
    shape = net_input_size, net_input_size

    # https://discuss.pytorch.org/t/how-can-l-load-my-best-model-as-a-feature-extractor-evaluator/17254/6
    activation = {}
    def get_activation(name):
        def hook(model, input, output):
            activation[name] = output.detach()
        return hook

    # Actual network
    net = LeNet(shape)
    net.fc2.register_forward_hook(get_activation('fc2'))
    net.conv2.register_forward_hook(get_activation('conv2'))

    # Log architecture
    dummy_input = torch.zeros(1, 1, net_input_size, net_input_size)
    writer.add_graph(
        net,
        input_to_model=dummy_input,
        # verbose=True,
    )

    # Loss
    criterion = nn.CrossEntropyLoss()

    # Optimizer
    optimizer = optim.SGD(
        net.parameters(),
        lr=0.001,
        momentum=0.9,
    )

    print('Training...')
    epochs = 41
    log_every_n_epochs = 10
    progress = tqdm(range(epochs))
    iteration = 0
    features_shape = len(dataset), net.fc2.out_features
    for epoch in progress:  # loop over the dataset multiple times
        features = np.empty(features_shape)
        epoch_labels = []

        # Log epoch validation loss
        net.eval()
        validation_loss = 0
        for batch_index, batch in enumerate(validation_loader):
            inputs = batch['image']
            labels = batch['target']
            outputs = net(inputs)
            batch_loss = criterion(outputs, labels)
            validation_loss += batch_loss.item()
        num_validation_batches = len(validation_loader)
        validation_loss /= num_validation_batches
        writer.add_scalars(
            'Loss',
            {'Epoch/validation': validation_loss},
            iteration,
        )


        # Log stuff every N epochs
        if epoch % log_every_n_epochs == 0 and verbose:
            # Log embedding of last layer before classifier
            for batch_index, batch in enumerate(validation_loader):
                inputs = batch['image']
                labels = batch['target']
                epoch_labels += labels.tolist()
                outputs = net(inputs)
                ini = batch_index * batch_size
                this_batch_size = inputs.shape[0]
                fin = ini + this_batch_size
                features_tensor = activation['fc2']
                features[ini:fin] = features_tensor.numpy()
            indices = np.array(epoch_labels)
            epoch_labels = np.array(dataset.classes)[indices]
            writer.add_embedding(
                features,
                global_step=iteration,
                tag='Validation/Embedding',
                metadata=epoch_labels,
            )

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
            inputs = batch['image']
            labels = batch['target']
            sampled_images.append(batch['name'])

            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = net(inputs)

            # Compute loss
            batch_loss = criterion(outputs, labels)

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

            # Log input images
            inputs_numpy = inputs.expand(-1, 3, -1, -1).numpy()
            inputs_numpy -= inputs_numpy.min()
            inputs_numpy /= inputs_numpy.max()
            writer.add_images('Training/Input', inputs_numpy, iteration)

            # Log activations of conv2
            if verbose:
                batch_activations = activation['conv2']
                for filter_idx in range(batch_activations.shape[1]):
                    layer_activations = batch_activations[:, filter_idx:filter_idx+1, :, :]
                    layer_activations = layer_activations.expand(-1, 3, -1, -1)
                    layer_activations_np = layer_activations.numpy()
                    layer_activations_np -= layer_activations_np.min()
                    layer_activations_np /= layer_activations_np.max()
                    writer.add_images(
                        f'Conv2/activation {filter_idx}',
                        layer_activations_np,
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

    lines = []
    for iteration, images_names in enumerate(sampled_images):
        lines.append(f'Iteration {iteration: 5d}: {", ".join(images_names)}')

    log_path.write_text('\n'.join(lines))

    print()
    print('Finished Training')
