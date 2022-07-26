import logging
import sys
from glob import glob
from tqdm import tqdm
import numpy as np
import pandas as pd
import cv2

sys.path.append('..')
import settings
import annotation_utils


if __name__ == '__main__':

    dataset_path = settings.DATA / 'external_data' / 'HuBMAP_Colonic_Crypt'
    image_filenames = sorted(glob(str(dataset_path / 'images' / '*.tiff')))

    metadata = []

    for image_filename in tqdm(image_filenames):

        image_id = image_filename.split("/")[-1].split(".")[0]
        mask_filename = str(dataset_path / 'masks' / f'{image_id}.tiff')

        image = cv2.imread(image_filename)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # Extract metadata from image
        image_r_mean = np.mean(image[:, :, 0])
        image_r_std = np.std(image[:, :, 0])
        image_g_mean = np.mean(image[:, :, 1])
        image_g_std = np.std(image[:, :, 1])
        image_b_mean = np.mean(image[:, :, 2])
        image_b_std = np.std(image[:, :, 2])

        mask = cv2.imread(mask_filename, -1)
        # Extract metadata from mask
        mask_area = np.sum(mask)

        metadata.append({
            'id': image_id,
            'organ': 'colon',
            'data_source': 'HuBMAP_Colonic_Crypt',
            'stain': 'H&E',
            'image_height': image.shape[0],
            'image_width': image.shape[1],
            'pixel_size': np.nan,
            'tissue_thickness': np.nan,
            'rle': annotation_utils.encode_rle_mask(mask),
            'age': np.nan,
            'sex': np.nan,
            'image_r_mean': image_r_mean,
            'image_r_std': image_r_std,
            'image_g_mean': image_g_mean,
            'image_g_std': image_g_std,
            'image_b_mean': image_b_mean,
            'image_b_std': image_b_std,
            'mask_area': mask_area,
            'image_filename': image_filename,
            'mask_filename': mask_filename
        })

    df_metadata = pd.DataFrame(metadata)
    df_metadata.to_csv(dataset_path / 'metadata.csv', index=False)
    logging.info(f'Saved metadata.csv to {dataset_path}')
    logging.info(f'Dataset Shape: {df_metadata.shape} - Memory Usage: {df_metadata.memory_usage().sum() / 1024 ** 2:.2f} MB')
