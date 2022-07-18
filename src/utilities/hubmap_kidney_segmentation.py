import logging
import sys
from pathlib import Path
from glob import glob
from tqdm import tqdm
import numpy as np
import pandas as pd
import cv2
from torch.utils.data import Dataset
import rasterio
from rasterio.windows import Window

sys.path.append('..')
import settings
import annotation_utils


tile_size = 1024
reduction_factor = 4
saturation_threshold = 40
pixel_threshold = 1000 * (tile_size // 256) ** 2


class HuBMAPDataset(Dataset):

    def __init__(self, image_filename, rle=None, tile_size=1024, reduction_factor=4):

        self.image = rasterio.open(image_filename, num_threads='all_cpus')

        if self.image.count != 3:
            subdatasets = self.image.subdatasets
            self.layers = []
            if len(subdatasets) > 0:
                for i, subdataset in enumerate(subdatasets, 0):
                    self.layers.append(rasterio.open(subdataset))

        self.shape = self.image.shape
        self.reduction_factor = reduction_factor
        self.tile_size = reduction_factor * tile_size
        self.pad0 = (self.tile_size - self.shape[0] % self.tile_size) % self.tile_size
        self.pad1 = (self.tile_size - self.shape[1] % self.tile_size) % self.tile_size
        self.n0max = (self.shape[0] + self.pad0) // self.tile_size
        self.n1max = (self.shape[1] + self.pad1) // self.tile_size
        self.mask = annotation_utils.decode_rle_mask(rle, (self.shape[1], self.shape[0])) if rle is not None else None

    def __len__(self):
        return self.n0max * self.n1max

    def __getitem__(self, idx):

        n0, n1 = idx // self.n1max, idx % self.n1max
        x0, y0 = -self.pad0 // 2 + n0 * self.tile_size, -self.pad1 // 2 + n1 * self.tile_size

        p00, p01 = max(0, x0), min(x0 + self.tile_size, self.shape[0])
        p10, p11 = max(0, y0), min(y0 + self.tile_size, self.shape[1])
        img = np.zeros((self.tile_size, self.tile_size, 3), np.uint8)
        mask = np.zeros((self.tile_size, self.tile_size), np.uint8)

        if self.image.count == 3:
            img[(p00 - x0):(p01 - x0), (p10 - y0):(p11 - y0)] = np.moveaxis(self.image.read([1, 2, 3], window=Window.from_slices((p00, p01), (p10, p11))), 0, -1)
        else:
            for i, layer in enumerate(self.layers):
                img[(p00 - x0):(p01 - x0), (p10 - y0):(p11 - y0), i] = layer.read(1, window=Window.from_slices((p00, p01), (p10, p11)))
        if self.mask is not None:
            mask[(p00 - x0):(p01 - x0), (p10 - y0):(p11 - y0)] = self.mask[p00:p01, p10:p11]

        if self.reduction_factor != 1:
            img = cv2.resize(img, (self.tile_size // reduction_factor, self.tile_size // reduction_factor), interpolation=cv2.INTER_AREA)
            mask = cv2.resize(mask, (self.tile_size // reduction_factor, self.tile_size // reduction_factor), interpolation=cv2.INTER_NEAREST)

        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)

        return img, mask, (-1 if (s > saturation_threshold).sum() <= pixel_threshold or img.sum() <= pixel_threshold else idx)


if __name__ == '__main__':

    # Create directory for epoch predictions visualizations
    tiled_hubmap_kidney_segmentation_image_directory = Path(settings.DATA / 'tiled_hubmap_kidney_segmentation_images')
    tiled_hubmap_kidney_segmentation_image_directory.mkdir(parents=True, exist_ok=True)

    df_metadata = pd.read_csv(settings.DATA / 'hubmap_kidney_segmentation' / 'HuBMAP-20-dataset_information.csv')
    df_metadata['id'] = df_metadata['image_file'].apply(lambda x: str(x).split('.')[0])
    df_train = pd.read_csv(settings.DATA / 'hubmap_kidney_segmentation' / 'train.csv')
    df_metadata = df_metadata.merge(df_train, on='id', how='left')
    df_metadata = df_metadata.loc[df_metadata['encoding'].notna(), :].reset_index(drop=True)
    df_metadata['image_filename'] = df_metadata['image_file'].apply(lambda image_file: str(settings.DATA / 'hubmap_kidney_segmentation' / 'train' / image_file))
    train_images = glob(str(settings.DATA / 'hubmap_kidney_segmentation' / 'train' / '*.tiff'))

    for image_idx, row in tqdm(df_metadata.iterrows(), total=len(df_metadata)):

        if isinstance(row['encoding'], str) is False:
            continue

        dataset = HuBMAPDataset(
            image_filename=row['image_filename'],
            rle=row['encoding'],
            tile_size=tile_size,
            reduction_factor=reduction_factor
        )

        logging.info(f'Creating tile images and masks from {row["id"]}')

        for dataset_idx in tqdm(range(len(dataset))):

            try:
                tile_image, tile_mask, tile_idx = dataset[dataset_idx]
                if tile_idx < 0:
                    continue

                cv2.imwrite(str(tiled_hubmap_kidney_segmentation_image_directory / f'{row["id"]}_{tile_idx}.png'), tile_image)
                np.save(str(tiled_hubmap_kidney_segmentation_image_directory / f'{row["id"]}_{tile_idx}.npy'), tile_mask)

            except ValueError:
                continue
