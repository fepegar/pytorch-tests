from pathlib import Path
from collections import namedtuple

import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision.transforms import Pad, Resize, Compose


class ResectionSideDataset(Dataset):
    def __init__(self, root_dir, padding, resize=64):
        self.root_dir = Path(root_dir)
        self.images_dir = self.root_dir / 'images'
        self.labels_dir = self.root_dir / 'labels'
        self.images_filepaths = sorted(list(self.images_dir.glob('*/*tif')))
        self.labels_filepaths = sorted(list(self.labels_dir.glob('*/*tif')))
        self.classes = (
            'left',
            'right',
        )
        self.image_transform = self.get_transform(
            padding, resize, Image.LANCZOS)
        self.label_transform = self.get_transform(
            padding, resize, Image.NEAREST)
        sample_fields = [
            'image',
            'label',
            'laterality_target',
            'name',
            'bounding_box',
        ]
        self.sample_tuple = namedtuple('Sample', sample_fields)

    def __len__(self):
        return len(self.images_filepaths)

    def __getitem__(self, idx):
        image_path = self.images_filepaths[idx]
        image = Image.open(image_path)
        image = self.image_transform(image)
        array = np.array(image)
        image_tensor = torch.from_numpy(array)  # height x width
        image_tensor = torch.unsqueeze(
            image_tensor, 0)  # channels x height x width

        label_path = self.labels_filepaths[idx]
        label_image = Image.open(label_path)
        label_image = self.label_transform(label_image)
        label_array = np.array(label_image)
        label_tensor = torch.from_numpy(label_array)  # height x width
        label_tensor = torch.unsqueeze(
            label_tensor, 0)  # channels x height x width
        # pylint: disable=not-callable
        bounding_box = torch.tensor(self.get_bounding_box(label_array))

        laterality = image_path.parent.name
        laterality_target = self.classes.index(laterality)

        sample = self.sample_tuple(
            image=image_tensor,
            label=label_tensor,
            laterality_target=laterality_target,
            name=image_path.stem,
            bounding_box=bounding_box,
        )
        return sample

    def get_transform(self, padding, size, interpolation):
        return Compose([
            Pad(padding, padding_mode='edge'),
            Resize((size, size), interpolation),
        ])

    def get_bounding_box(self, array):
        """
        Return normalized coordinates of bounding box as:
        (center_y, center_x, height, width)
        """
        coords = np.array(np.where(array)).T  # N x 2
        shape = np.array(array.shape)
        cmin = coords.min(axis=0) - 0.5
        cmax = coords.max(axis=0) + 0.5
        cmin = cmin / shape  # normalize
        cmax = cmax / shape  # normalize
        center_y, center_x = (cmin + cmax) / 2
        height, width = cmax - cmin
        return center_y, center_x, height, width
