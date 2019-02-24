"""
For each subject, I need:
- Postop on MNI
- Head mask MNI
- Resection mask MNI
- Side
"""

from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm
from utils import nifti

from episurg.utils import get_all_subjects
from episurg.image import get_centroid_slices


if __name__ == '__main__':
    output_dir = Path('/tmp/pytorch_dataset')
    output_dir.mkdir(exist_ok=True, parents=True)
    subjects = [
        subject
        for subject
        in get_all_subjects()
        if subject.id.startswith('rm')
    ]

    sides = {}

    for subject in tqdm(subjects):
        tiff_path = output_dir / f'{subject.id}.tiff'
        if not tiff_path.is_file():
            nii_path = subject.t1_post.whitened_image.path
            nii_seg_path = subject.t1_post.resection_on_mni_path
            _, _, k = get_centroid_slices(nii_seg_path)
            nii = nifti.load(nii_path)
            nii_slice = nii.dataobj[..., k]
            image = Image.fromarray(nii_slice)
            image.save(tiff_path)

            nii_slice = nii_slice - nii_slice.min()
            nii_slice /= nii_slice.max()
            nii_slice *= 255
            nii_slice = nii_slice.astype(np.uint8)
            image = Image.fromarray(nii_slice)
            png_path = output_dir / f'{subject.id}.png'
            image.save(png_path)

        time = datetime.fromtimestamp(subject.dir.stat().st_mtime)
        if time.month == 1 and time.day == 10:
            side = 'right'
        elif time.month == 12 and time.day == 17:
            side = 'left'
        else:
            side = 'left'  # for rm0056

        side_dir = output_dir / side
        side_dir.mkdir(exist_ok=True, parents=True)
        tiff_path.rename(side_dir / f'{subject.id}.tiff')

    df = pd.DataFrame.from_dict(sides, orient='index', columns=('laterality',))
    df.loc['rm0056.tiff'] = 1
    df.to_csv(output_dir / 'data.csv')
