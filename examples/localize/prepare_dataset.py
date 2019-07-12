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

import episurg.scan.mni as mni
from episurg.utils import get_all_subjects
from episurg.image import get_centroid_slices


def get_largest_axial_slice(image_path):
    data = nifti.load(image_path).get_data()
    max_area = -1
    max_index = None
    for k in range(data.shape[-1]):
        area = np.count_nonzero(data[..., k])
        if area > max_area:
            max_area = area
            max_index = k

    return max_index



if __name__ == '__main__':
    output_dir = Path('/tmp/pytorch_dataset')
    if output_dir.is_dir():
        import shutil
        shutil.rmtree(output_dir)

    images_dir = output_dir / 'images'
    labels_dir = output_dir / 'labels'
    masks_dir = output_dir / 'masks'

    images_dir.mkdir(exist_ok=True, parents=True)
    labels_dir.mkdir(exist_ok=True, parents=True)
    masks_dir.mkdir(exist_ok=True, parents=True)

    subjects = [
        subject
        for subject
        in get_all_subjects()
        if subject.id.startswith('rm')  # Use only Daichi's
    ]

    sides = {}

    mni_path = mni.MNI().brain_mask_path
    mni_nii = nifti.load(mni_path)

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
        masks_side_dir = masks_dir / side
        masks_side_dir.mkdir(exist_ok=True, parents=True)

        tiff_image_path = images_side_dir / f'{subject.id}.tif'

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

            # PNG preview
            nii_slice = nii_slice * 255
            image = Image.fromarray(nii_slice)
            png_path = labels_side_dir / f'{subject.id}.png'
            image.save(png_path)

            # Make masks images
            mask_slice = mni_nii.dataobj[..., k] * 255
            mask = Image.fromarray(mask_slice)
            png_path = masks_side_dir / f'{subject.id}.png'
            mask.save(png_path)
