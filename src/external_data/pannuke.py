import logging
import sys
from tqdm import tqdm
import numpy as np
import pandas as pd
import cv2

sys.path.append('..')
import settings
import annotation_utils


if __name__ == '__main__':

    dataset_path = settings.DATA / 'external_data' / 'PanNuke'
    (dataset_path / 'images').mkdir(parents=True, exist_ok=True)
    (dataset_path / 'masks').mkdir(parents=True, exist_ok=True)

    metadata = []

    for fold in range(1, 4):

        types = np.load(dataset_path / f'types{fold}.npy')
        images = np.load(dataset_path / f'images{fold}.npy').astype(np.uint8)
        masks = np.any(np.load(dataset_path / f'masks{fold}.npy').astype(np.uint8), axis=3).astype(np.uint8)

        logging.info(f'Images Shape: {images.shape} - Memory Usage: {images.nbytes / 1024 ** 2:.2f} MB')
        logging.info(f'Masks Shape: {masks.shape} - Memory Usage: {masks.nbytes / 1024 ** 2:.2f} MB')
        logging.info(f'Types Shape: {types.shape} - Memory Usage: {types.nbytes / 1024 ** 2:.2f} MB')

        for idx in tqdm(range(images.shape[0])):

            organ = types[idx]
            if organ not in ['Kidney', 'Lung', 'Prostate', 'Colon']:
                continue
            else:
                image = images[idx]
                mask = masks[idx]

                # Extract metadata from image
                image_r_mean = np.mean(image[:, :, 0])
                image_r_std = np.std(image[:, :, 0])
                image_g_mean = np.mean(image[:, :, 1])
                image_g_std = np.std(image[:, :, 1])
                image_b_mean = np.mean(image[:, :, 2])
                image_b_std = np.std(image[:, :, 2])

                # Extract metadata from mask
                mask_area = np.sum(mask)

                image_filename = str(dataset_path / 'images' / f'image{idx}_fold{fold}_{organ}.png')
                mask_filename = str(dataset_path / 'masks' / f'image{idx}_fold{fold}_{organ}.png')

                metadata.append({
                    'id': f'image{idx}_fold{fold}_{organ}',
                    'organ': organ,
                    'data_source': 'PanNuke',
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

                cv2.imwrite(image_filename, image)
                cv2.imwrite(mask_filename, mask)

        del images, masks, types

    df_metadata = pd.DataFrame(metadata)
    df_metadata.to_csv(dataset_path / 'metadata.csv', index=False)
    logging.info(f'Saved metadata.csv to {dataset_path}')
    logging.info(f'Dataset Shape: {df_metadata.shape} - Memory Usage: {df_metadata.memory_usage().sum() / 1024 ** 2:.2f} MB')
