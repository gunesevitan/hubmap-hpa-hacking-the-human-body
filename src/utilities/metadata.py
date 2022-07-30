import logging
import sys
from tqdm import tqdm
from glob import glob
import json
import numpy as np
import pandas as pd
import tifffile

sys.path.append('..')
import settings
import annotation_utils


if __name__ == '__main__':

    df_train = pd.read_csv(settings.DATA / 'train.csv')
    df_test = pd.read_csv(settings.DATA / 'test.csv')
    logging.info(f'Training Set Shape: {df_train.shape} - Memory Usage: {df_train.memory_usage().sum() / 1024 ** 2:.2f} MB')
    logging.info(f'Test Set Shape: {df_test.shape} - Memory Usage: {df_test.memory_usage().sum() / 1024 ** 2:.2f} MB')

    train_image_filenames = glob(str(settings.DATA / 'train_images' / '*.tiff'))
    test_image_filenames = glob(str(settings.DATA / 'test_images' / '*.tiff'))

    for image_filename in tqdm(train_image_filenames):

        image_id = int(image_filename.split('/')[-1].split('.')[0])

        # Extract metadata from image
        image = tifffile.imread(image_filename)

        df_train.loc[df_train['id'] == image_id, 'image_r_mean'] = np.mean(image[:, :, 0])
        df_train.loc[df_train['id'] == image_id, 'image_r_std'] = np.std(image[:, :, 0])
        df_train.loc[df_train['id'] == image_id, 'image_g_mean'] = np.mean(image[:, :, 1])
        df_train.loc[df_train['id'] == image_id, 'image_g_std'] = np.std(image[:, :, 1])
        df_train.loc[df_train['id'] == image_id, 'image_b_mean'] = np.mean(image[:, :, 2])
        df_train.loc[df_train['id'] == image_id, 'image_b_std'] = np.std(image[:, :, 2])

        # Extract metadata from mask
        rle_mask = df_train.loc[df_train['id'] == image_id, 'rle'].values[0]
        mask = annotation_utils.decode_rle_mask(rle_mask=rle_mask, shape=image.shape[:2]).T
        df_train.loc[df_train['id'] == image_id, 'mask_area'] = np.sum(mask)

        # Extract metadata from polygons
        polygons = annotation_utils.get_segmentation_polygons_from_json(
            annotation_directory='train_annotations',
            image_id=image_id
        )
        df_train.loc[df_train['id'] == image_id, 'n_polygons'] = len(polygons)

        df_train.loc[df_train['id'] == image_id, 'image_filename'] = image_filename
        df_train.loc[df_train['id'] == image_id, 'polygon_filename'] = settings.DATA / 'train_annotations' / f'{image_id}.json'
        df_train.loc[df_train['id'] == image_id, 'stain'] = 'DAB&H'

    df_train.rename(columns={'img_height': 'image_height', 'img_width': 'image_width'}, inplace=True)
    df_train.to_csv(settings.DATA / 'train_metadata.csv', index=False)
    logging.info(f'Saved train_metadata.csv to {settings.DATA}')
    logging.info(f'Training Set Shape: {df_train.shape} - Memory Usage: {df_train.memory_usage().sum() / 1024 ** 2:.2f} MB')

    for image_filename in tqdm(test_image_filenames):

        image_id = int(image_filename.split('/')[-1].split('.')[0])
        mask_filename = str(settings.DATA / 'test_masks' / f'{image_id}.npy')
        # Extract metadata from image
        image = tifffile.imread(image_filename)

        df_test.loc[df_test['id'] == image_id, 'image_r_mean'] = np.mean(image[:, :, 0])
        df_test.loc[df_test['id'] == image_id, 'image_r_std'] = np.std(image[:, :, 0])
        df_test.loc[df_test['id'] == image_id, 'image_g_mean'] = np.mean(image[:, :, 1])
        df_test.loc[df_test['id'] == image_id, 'image_g_std'] = np.std(image[:, :, 1])
        df_test.loc[df_test['id'] == image_id, 'image_b_mean'] = np.mean(image[:, :, 2])
        df_test.loc[df_test['id'] == image_id, 'image_b_std'] = np.std(image[:, :, 2])

        # Extract metadata from mask
        mask = np.load(mask_filename)
        df_test.loc[df_test['id'] == image_id, 'rle'] = annotation_utils.encode_rle_mask(mask.T)
        df_test.loc[df_test['id'] == image_id, 'mask_area'] = np.sum(mask)

        df_test.loc[df_test['id'] == image_id, 'image_filename'] = image_filename
        df_test.loc[df_test['id'] == image_id, 'mask_filename'] = mask_filename
        df_test.loc[df_test['id'] == image_id, 'stain'] = 'H&E'

    df_test['age'] = 0
    df_test['sex'] = 0
    df_test.rename(columns={'img_height': 'image_height', 'img_width': 'image_width'}, inplace=True)
    df_test.to_csv(settings.DATA / 'test_metadata.csv', index=False)
    logging.info(f'Saved test_metadata.csv to {settings.DATA}')
    logging.info(f'Test Set Shape: {df_test.shape} - Memory Usage: {df_test.memory_usage().sum() / 1024 ** 2:.2f} MB')

    imaging_measurements = {
        'hpa': {
            'pixel_size': {
                'kidney': 0.4,
                'prostate': 0.4,
                'largeintestine': 0.4,
                'spleen': 0.4,
                'lung': 0.4
            },
            'tissue_thickness': {
                'kidney': 4,
                'prostate': 4,
                'largeintestine': 4,
                'spleen': 4,
                'lung': 4
            }
        },
        'hubmap': {
            'pixel_size': {
                'kidney': 0.5,
                'prostate': 6.263,
                'largeintestine': 0.2290,
                'spleen': 0.4945,
                'lung': 0.7562
            },
            'tissue_thickness': {
                'kidney': 10,
                'prostate': 5,
                'largeintestine': 8,
                'spleen': 4,
                'lung': 5
            }
        }
    }
    with open(settings.DATA / 'imaging_measurements.json', mode='w') as f:
        json.dump(imaging_measurements, f, indent=2, ensure_ascii=False)
    logging.info(f'Saved imaging_measurements.json.csv to {settings.DATA}')
