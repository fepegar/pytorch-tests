from pathlib import Path
from collections import namedtuple

import numpy as np
from utils import nifti
import torch
from torch.utils.data import Dataset
from torchvision.transforms import Pad, Resize, Compose



class ResectionDataset(Dataset):
    def __init__(self, root_dir):
        self.root_dir = Path(root_dir)
        self.images_dir = self.root_dir / 'images'
        self.labels_dir = self.root_dir / 'labels'
        self.images_filepaths = sorted(list(self.images_dir.glob('*/*nii.gz')))
        self.labels_filepaths = sorted(list(self.labels_dir.glob('*/*nii.gz')))

        sample_fields = [
            'image',
            'label',
            'name',
            'bounding_box',
        ]
        self.sample_tuple = namedtuple('Sample', sample_fields)

    def __len__(self):
        return len(self.images_filepaths)

    def __getitem__(self, idx):
        image_path = self.images_filepaths[idx]
        image_array = nifti.load(image_path).get_data()
        image_tensor = torch.from_numpy(image_array)  # depth x height x width
        image_tensor = torch.unsqueeze(image_tensor, 0)  # channels x depth x height x width

        label_path = self.labels_filepaths[idx]
        label_array = nifti.load(label_path).get_data()
        label_tensor = torch.from_numpy(label_array)  # depth x height x width
        label_tensor = torch.unsqueeze(label_tensor, 0)  # channels x depth x height x width
        # pylint: disable=not-callable
        bounding_box = torch.tensor(self.get_bounding_box(label_array))

        sample = self.sample_tuple(
            image=image_tensor,
            label=label_tensor,
            name=image_path.stem,
            bounding_box=bounding_box,
        )
        return sample

    def get_bounding_box(self, array):
        """
        Return normalized coordinates of bounding box as:
        (center_i, center_j, center_k, size_i, size_j, size_k)
        """
        coords = np.array(np.where(array)).T  # N x 3
        shape = np.array(array.shape)
        cmin = coords.min(axis=0) - 0.5
        cmax = coords.max(axis=0) + 0.5
        cmin = cmin / shape  # normalize
        cmax = cmax / shape  # normalize
        center_i, center_j, center_k = (cmin + cmax) / 2
        size_i, size_j, size_k = cmax - cmin
        return center_i, center_j, center_k, size_i, size_j, size_k
