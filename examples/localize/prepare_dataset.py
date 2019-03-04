"""
For each subject, I need:
- Postop on MNI
- Resection mask MNI
- Side
"""

from pathlib import Path
from datetime import datetime

import numpy as np
from PIL import Image
from tqdm import tqdm
from utils import nifti

from episurg.utils import get_all_subjects
from episurg.image import get_centroid_slices


if __name__ == '__main__':
    output_dir = Path('/tmp/pytorch_dataset')
    images_dir = output_dir / 'images'
    labels_dir = output_dir / 'labels'
    images_dir.mkdir(exist_ok=True, parents=True)
    labels_dir.mkdir(exist_ok=True, parents=True)
    subjects = [
        subject
        for subject
        in get_all_subjects()
        if subject.id.startswith('rm')  # Use only Daichi's
    ]

    sides = {}

    for subject in tqdm(subjects):
        time = datetime.fromtimestamp(subject.dir.stat().st_mtime)
        if time.month == 1 and time.day == 10:
            side = 'right'
        elif time.month == 12 and time.day == 17:
            side = 'left'
        else:
            side = 'left'  # for rm0056
        images_side_dir = images_dir / side
        images_side_dir.mkdir(exist_ok=True, parents=True)
        labels_side_dir = labels_dir / side
        labels_side_dir.mkdir(exist_ok=True, parents=True)

        tiff_image_path = images_side_dir / f'{subject.id}.tif'
        tiff_label_path = labels_side_dir / f'{subject.id}.tif'
        if not tiff_image_path.is_file():
            # Make image slices
            nii_path = subject.t1_post.whitened_image.path
            nii_seg_path = subject.t1_post.resection_on_mni_path
            _, _, k = get_centroid_slices(nii_seg_path)
            nii = nifti.load(nii_path)
            nii_slice = nii.dataobj[..., k]
            image = Image.fromarray(nii_slice)
            image.save(tiff_image_path)

            # PNG preview
            nii_slice = nii_slice - nii_slice.min()
            nii_slice /= nii_slice.max()
            nii_slice *= 255
            nii_slice = nii_slice.astype(np.uint8)
            image = Image.fromarray(nii_slice)
            png_path = images_side_dir / f'{subject.id}.png'
            image.save(png_path)

            # Make segmentation slices
            nii = nifti.load(nii_seg_path)
            nii_slice = nii.dataobj[..., k]
            image = Image.fromarray(nii_slice)
            image.save(tiff_label_path)

            # PNG preview
            nii_slice = nii_slice * 255
            image = Image.fromarray(nii_slice)
            png_path = labels_side_dir / f'{subject.id}.png'
            image.save(png_path)
