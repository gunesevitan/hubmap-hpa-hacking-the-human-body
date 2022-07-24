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
    hpa_images_filenames = glob(str(settings.DATA / 'external_data' / 'HPA' / 'images' / '*.png'))
    hubmap_kidney_segmentation_images_filenames = glob(str(settings.DATA / 'external_data' / 'HuBMAP_Kidney_Segmentation' / 'images' / '*.png'))
    hubmap_portal_images_filenames = glob(str(settings.DATA / 'external_data' / 'HuBMAP_Portal' / 'images' / '*.png'))
    anhir_images_filenames = glob(str(settings.DATA / 'external_data' / 'ANHIR' / 'images' / '*.jpg'))
    consep_images_filenames = glob(str(settings.DATA / 'external_data' / 'CoNSeP' / 'images' / '*.png'))
    crag_images_filenames = glob(str(settings.DATA / 'external_data' / 'CRAG' / 'images' / '*.png'))
    crchistophenotypes_images_filenames = glob(str(settings.DATA / 'external_data' / 'CRCHistoPhenotypes' / 'images' / '*.bmp'))
    glas_images_filenames = glob(str(settings.DATA / 'external_data' / 'GlaS' / 'images' / '*.bmp'))
    lyon19_images_filenames = glob(str(settings.DATA / 'external_data' / 'LYON19' / 'images' / '*.png'))
    monusac_images_filenames = glob(str(settings.DATA / 'external_data' / 'MoNuSAC' / 'images' / '*.tif'))
    monuseg_images_filenames = glob(str(settings.DATA / 'external_data' / 'MoNuSeg' / 'images' / '*.tif'))
    pannuke_images_filenames = glob(str(settings.DATA / 'external_data' / 'PanNuke' / 'images' / '*.png'))

    image_filenames = train_raw_images_filenames + test_raw_images_filenames + hpa_images_filenames +\
                      hubmap_kidney_segmentation_images_filenames + hubmap_portal_images_filenames +\
                      anhir_images_filenames + consep_images_filenames + crag_images_filenames +\
                      crchistophenotypes_images_filenames + glas_images_filenames + lyon19_images_filenames +\
                      monusac_images_filenames + monuseg_images_filenames + pannuke_images_filenames

    for image_filename in tqdm(image_filenames):

        if image_filename.split('.')[-1] == 'tiff' or image_filename.split('.')[-1] == 'tif':
            image = tifffile.imread(image_filename)
        elif image_filename.split('.')[-1] == 'png' or image_filename.split('.')[-1] == 'jpg' or image_filename.split('.')[-1] == 'bmp':
            image = cv2.imread(image_filename)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if image.shape[2] == 4:
            image = image[:, :, :3]

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
