import logging
import sys
from glob import glob
import tifffile
import numpy as np
import pandas as pd
import cv2

sys.path.append('..')
import settings
import preprocessing


if __name__ == '__main__':

    hubmap_external_image_filenames = glob(str(settings.DATA / 'hubmap_external_images' / '*.tiff'))
    metadata = []

    for image_filename in sorted(hubmap_external_image_filenames):

        image = tifffile.imread(image_filename)
        processed_image_filename = '.'.join(image_filename.split("/")[-1].split(".")[:-1])
        processed_image_filename = f'{processed_image_filename}.png'

        if image.shape[0] == 3:
            image = np.moveaxis(image, 0, -1)

        image, _ = preprocessing.crop_image(
            image,
            mask=None,
            crop_black_border=True,
            crop_background=False,
            verbose=True
        )
        # Resize image to 4500x4500 since it is the largest HuBMAP image dimensions
        image = cv2.resize(image, (4500, 4500), cv2.INTER_NEAREST)
        cv2.imwrite(str(settings.DATA / 'hubmap_external_images' / processed_image_filename), image)
        logging.info(f'Processed image saved to {str(settings.DATA / "hubmap_external_images" / processed_image_filename)}')

        metadata.append({
            'id': processed_image_filename.split('_')[0],
            'stain': processed_image_filename.split('_')[1],
            'organ': processed_image_filename.split('_')[2],
            'pixel_size': float(processed_image_filename.split('_')[3]),
            'age': int(processed_image_filename.split('_')[0]),
            'sex': processed_image_filename.split('_')[0],
            'image_filename': processed_image_filename,
            'mask_filename': processed_image_filename.replace('.png', '.npy')
        })

    df_metadata = pd.DataFrame(metadata)
    df_metadata.to_csv(settings.DATA / 'hubmap_external_metadata.csv', index=False)
    logging.info(f'Saved hubmap_external_metadata.csv to {settings.DATA}')
