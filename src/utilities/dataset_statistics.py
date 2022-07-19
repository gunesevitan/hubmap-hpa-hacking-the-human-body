import logging
import sys
from tqdm import tqdm
from glob import glob
import json
import numpy as np
import cv2
import tifffile

sys.path.append('..')
import settings


if __name__ == '__main__':

    pixel_count = 0
    pixel_sum = 0
    pixel_squared_sum = 0

    train_raw_images_filenames = glob(str(settings.DATA / 'train_images' / '*.tiff'))
    test_raw_images_filenames = glob(str(settings.DATA / 'test_images' / '*.tiff'))
    hubmap_kidney_segmentation_images_filenames = glob(str(settings.DATA / 'hubmap_kidney_segmentation' / 'images' / '*.png'))

    image_filenames = train_raw_images_filenames + test_raw_images_filenames + hubmap_kidney_segmentation_images_filenames

    for image_filename in tqdm(image_filenames):

        if image_filename.split('.')[-1] == 'tiff':
            image = tifffile.imread(image_filename)
        elif image_filename.split('.')[-1] == 'png':
            image = cv2.imread(image_filename)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

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
    with open(settings.DATA / 'statistics.json', mode='w') as f:
        json.dump(dataset_statistics, f, indent=2)

    logging.info(f'Dataset statistics calculated with {len(image_filenames)} images and saved to {settings.DATA}')
