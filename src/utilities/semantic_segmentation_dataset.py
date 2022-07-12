import logging
import sys
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
    masks_path.mkdir(parents=True, exist_ok=True)

    for idx, row in tqdm(df_train.iterrows(), total=df_train.shape[0]):
        mask = annotation_utils.decode_rle_mask(
            rle_mask=row['rle'],
            shape=(row['img_height'], row['img_width'])
        ).T
        np.save(masks_path / f'{row["id"]}.npy', mask)

    logging.info(f'train_masks_raw dataset is created at {settings.DATA} with {df_train.shape[0]} masks')
