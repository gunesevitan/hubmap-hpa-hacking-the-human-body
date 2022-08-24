import logging
import sys
from glob import glob
from tqdm import tqdm
import numpy as np
import pandas as pd

sys.path.append('..')
import settings
import annotation_utils


if __name__ == '__main__':

    df_train = pd.read_csv(settings.DATA / 'train_metadata.csv')
    logging.info(f'Training Set Shape: {df_train.shape} - Memory Usage: {df_train.memory_usage().sum() / 1024 ** 2:.2f} MB')

    masks_path = settings.DATA / 'train_masks_raw'
    mask_filenames = glob(str(masks_path / '*.npy'))

    for mask_filename in tqdm(mask_filenames):

        image_id = int(mask_filename.split('/')[-1].split('.')[0])
        mask = np.load(mask_filename)
        df_train.loc[df_train['id'] == image_id, 'rle_corrected'] = annotation_utils.encode_rle_mask(mask.T)

    df_train.to_csv(settings.DATA / 'train_metadata.csv', index=False)
