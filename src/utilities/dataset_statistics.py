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


if __name__ == '__main__':

    df_train = pd.read_csv(settings.DATA / 'train_metadata.csv')
    logging.info(f'Training Set Shape: {df_train.shape} - Memory Usage: {df_train.memory_usage().sum() / 1024 ** 2:.2f} MB')

    pixel_count = 0
    pixel_sum = 0
    pixel_squared_sum = 0

    train_raw_images_filenames = glob(str(settings.DATA / 'train_images' / '*.tiff'))
    test_raw_images_filenames = glob(str(settings.DATA / 'test_images' / '*.tiff'))
    raw_images_filenames = train_raw_images_filenames + test_raw_images_filenames

    for image_filename in tqdm(raw_images_filenames):

        image = tifffile.imread(image_filename)
        image = np.float32(image) / 255.

        # Accumulate pixel counts, sums and squared sums for dataset mean and standard deviation computation
        pixel_count += (image.shape[0] * image.shape[1])
        pixel_sum += np.sum(image, axis=(0, 1))
        pixel_squared_sum += np.sum(image ** 2, axis=(0, 1))

    mean = pixel_sum / pixel_count
    var = (pixel_squared_sum / pixel_count) - (mean ** 2)
    std = np.sqrt(var)

    # Save dataset statistics as a json file
    dataset_statistics = {
        'mean': mean.tolist(),
        'std': std.tolist()
    }
    with open(settings.DATA / 'raw_images_statistics.json', mode='w') as f:
        json.dump(dataset_statistics, f, indent=2)

    logging.info(f'Raw images statistics calculated with {len(raw_images_filenames)} images and saved to {settings.DATA}')
