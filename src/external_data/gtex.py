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
import preprocessing


if __name__ == '__main__':

    # JPG files are very large since their pixel sizes are quite small
    # Prostate images doesn't need any further resize operation
    image_sizes = {
        'spleen': 2400,
        'prostate': None,
    }

    pixel_sizes = {
        'spleen': 0.4945,
        'prostate': 6.263
    }

    resize_images = True
    dataset_path = settings.DATA / 'external_data' / 'GTEx'
    raw_image_filenames = sorted(glob(str(dataset_path / 'raw_images' / '*')))

    if resize_images:
        for image_filename in tqdm(raw_image_filenames):

            image_id = image_filename.split("/")[-1].split(".")[0]
            image_format = image_filename.split("/")[-1].split(".")[-1]
            organ = image_id.split('_')[-2]
            image = cv2.imread(image_filename)

            if resize_images:
                longest_edge = image_sizes[organ]
                if longest_edge is not None:
                    image = preprocessing.resize_with_aspect_ratio(image=image, longest_edge=longest_edge)

                cv2.imwrite(str(dataset_path / 'images' / f'{image_id}.{image_format}'), image)

    image_filenames = sorted(glob(str(dataset_path / 'images' / '*')))

    metadata = []

    for image_filename in tqdm(image_filenames):

        image_id = image_filename.split("/")[-1].split(".")[0]
        mask_filename = str(dataset_path / 'masks' / f'{image_id}.npy')

        image = cv2.imread(image_filename)
        organ = image_id.split('_')[-2]

        # Extract metadata from image
        image_r_mean = np.mean(image[:, :, 0])
        image_r_std = np.std(image[:, :, 0])
        image_g_mean = np.mean(image[:, :, 1])
        image_g_std = np.std(image[:, :, 1])
        image_b_mean = np.mean(image[:, :, 2])
        image_b_std = np.std(image[:, :, 2])

        try:
            mask = np.load(mask_filename)
            # Extract metadata from mask
            rle = annotation_utils.encode_rle_mask(mask)
            mask_area = np.sum(mask)
        except FileNotFoundError:
            mask_filename = np.nan
            rle = np.nan
            mask_area = np.nan

        metadata.append({
            'id': image_id,
            'organ': organ,
            'data_source': 'GTEx',
            'stain': 'H&E',
            'image_height': image.shape[0],
            'image_width': image.shape[1],
            'pixel_size': pixel_sizes[organ],
            'tissue_thickness': np.nan,
            'rle': rle,
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
