from pathlib import Path

import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset


class ResectionSideDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = Path(root_dir)
        self.filepaths = list(self.root_dir.glob('*/*tif'))
        self.transform = transform
        self.classes = (
            'left',
            'right',
        )

    def __len__(self):
        return len(self.filepaths)

    def __getitem__(self, idx):
        image_path = self.filepaths[idx]
        image = Image.open(image_path)

        if self.transform is not None:
            image = self.transform(image)

        array = np.array(image)
        tensor = torch.from_numpy(array)  # height x width
        tensor = torch.unsqueeze(tensor, 0)  # channels x height x width
        laterality = image_path.parent.name
        target = self.classes.index(laterality)

        sample = {
            'image': tensor,
            'target': target,
            'name': image_path.name,
        }

        return sample
