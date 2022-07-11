import logging
import sys
from tqdm import tqdm
from glob import glob
import numpy as np
import pandas as pd
import cv2

sys.path.append('..')
import settings
import annotation_utils


if __name__ == '__main__':

    df_train = pd.read_csv(settings.DATA / 'train.csv')
    logging.info(f'Training Set Shape: {df_train.shape} - Memory Usage: {df_train.memory_usage().sum() / 1024 ** 2:.2f} MB')

    image_filenames = glob(str(settings.DATA / 'train_images' / '*.tiff'))

    for image_filename in tqdm(image_filenames):

        # Extract metadata from image
        image_id = int(image_filename.split('/')[-1].split('.')[0])
        image = cv2.imread(image_filename, -1)

        df_train.loc[df_train['id'] == image_id, 'image_r_mean'] = np.mean(image[:, :, 0])
        df_train.loc[df_train['id'] == image_id, 'image_r_std'] = np.std(image[:, :, 0])
        df_train.loc[df_train['id'] == image_id, 'image_g_mean'] = np.mean(image[:, :, 1])
        df_train.loc[df_train['id'] == image_id, 'image_g_std'] = np.std(image[:, :, 1])
        df_train.loc[df_train['id'] == image_id, 'image_b_mean'] = np.mean(image[:, :, 2])
        df_train.loc[df_train['id'] == image_id, 'image_b_std'] = np.std(image[:, :, 2])

        # Extract metadata from mask
        rle_mask = df_train.loc[df_train['id'] == image_id, 'rle'].values[0]
        mask = annotation_utils.decode_rle_mask(rle_mask=rle_mask, shape=image.shape[:2]).T
        df_train.loc[df_train['id'] == image_id, 'rle_mask_area'] = np.sum(mask)

        # Extract metadata from polygons
        polygons = annotation_utils.get_segmentation_polygons_from_json(
            annotation_directory='train_annotations',
            image_id=image_id
        )
        df_train.loc[df_train['id'] == image_id, 'n_polygons'] = len(polygons)

    df_train.to_csv(settings.DATA / 'train_metadata.csv', index=False)
    logging.info(f'Saved train_metadata.csv to {settings.DATA}')
    logging.info(f'Training Set Shape: {df_train.shape} - Memory Usage: {df_train.memory_usage().sum() / 1024 ** 2:.2f} MB')
