import logging
import sys
from glob import glob
from tqdm import tqdm
import xml.etree.ElementTree as ET
import numpy as np
import pandas as pd
import tifffile

sys.path.append('..')
import settings


organs = {
    'TCGA-A7-A13E-01Z-00-DX1': 'breast',
    'TCGA-A7-A13F-01Z-00-DX1': 'breast',
    'TCGA-AR-A1AK-01Z-00-DX1': 'breast',
    'TCGA-AR-A1AS-01Z-00-DX1': 'breast',
    'TCGA-E2-A1B5-01Z-00-DX1': 'breast',
    'TCGA-E2-A14V-01Z-00-DX1': 'breast',
    'TCGA-AC-A2FO-01A-01-TS1': 'breast',
    'TCGA-AO-A0J2-01A-01-BSA': 'breast',
    'TCGA-B0-5711-01Z-00-DX1': 'kidney',
    'TCGA-HE-7128-01Z-00-DX1': 'kidney',
    'TCGA-HE-7129-01Z-00-DX1': 'kidney',
    'TCGA-HE-7130-01Z-00-DX1': 'kidney',
    'TCGA-B0-5710-01Z-00-DX1': 'kidney',
    'TCGA-2Z-A9J9-01A-01-TS1': 'kidney',
    'TCGA-GL-6846-01A-01-BS1': 'kidney',
    'TCGA-IZ-8196-01A-01-BS1': 'kidney',
    'TCGA-B0-5698-01Z-00-DX1': 'liver',
    'TCGA-18-5592-01Z-00-DX1': 'liver',
    'TCGA-38-6178-01Z-00-DX1': 'liver',
    'TCGA-49-4488-01Z-00-DX1': 'liver',
    'TCGA-50-5931-01Z-00-DX1': 'liver',
    'TCGA-21-5784-01Z-00-DX1': 'liver',
    'TCGA-21-5786-01Z-00-DX1': 'liver',
    'TCGA-G9-6336-01Z-00-DX1': 'prostate',
    'TCGA-G9-6348-01Z-00-DX1': 'prostate',
    'TCGA-G9-6356-01Z-00-DX1': 'prostate',
    'TCGA-G9-6363-01Z-00-DX1': 'prostate',
    'TCGA-CH-5767-01Z-00-DX1': 'prostate',
    'TCGA-G9-6362-01Z-00-DX1': 'prostate',
    'TCGA-EJ-A46H-01A-03-TSC': 'prostate',
    'TCGA-HC-7209-01A-01-TS1': 'prostate',
    'TCGA-DK-A2I6-01A-01-TS1': 'bladder',
    'TCGA-G2-A2EK-01A-02-TSB': 'bladder',
    'TCGA-CU-A0YN-01A-02-BSB': 'bladder',
    'TCGA-ZF-A9R5-01A-01-TS1': 'bladder',
    'TCGA-AY-A8YK-01A-01-TS1': 'colon',
    'TCGA-NH-A8F7-01A-01-TS1': 'colon',
    'TCGA-A6-6782-01A-01-BS1': 'colon',
    'TCGA-KB-A93J-01A-01-TS1': 'stomach',
    'TCGA-RD-A8N9-01A-01-TS1': 'stomach',
    'TCGA-44-2665-01B-06-BS6': 'lung',
    'TCGA-69-7764-01A-01-TS1': 'lung',
    'TCGA-FG-A4MU-01B-01-TS1': 'brain',
    'TCGA-HT-8564-01Z-00-DX1': 'brain'
}


if __name__ == '__main__':

    dataset_path = settings.DATA / 'external_data' / 'MoNuSeg'
    image_filenames = glob(str(dataset_path / 'images' / '*.tif'))

    metadata = []

    for image_filename in tqdm(image_filenames):

        image_id = image_filename.split("/")[-1].split(".")[0]
        annotations_filename = str(dataset_path / 'annotations' / f'{image_id}.xml')
        annotations = ET.parse(annotations_filename).getroot()

        try:
            pixel_size = float(annotations.attrib['MicronsPerPixel'])
        except KeyError:
            pixel_size = np.nan

        try:
            organ = organs[image_id]
        except KeyError:
            organ = np.nan

        image = tifffile.imread(image_filename)
        # Extract metadata from image
        image_r_mean = np.mean(image[:, :, 0])
        image_r_std = np.std(image[:, :, 0])
        image_g_mean = np.mean(image[:, :, 1])
        image_g_std = np.std(image[:, :, 1])
        image_b_mean = np.mean(image[:, :, 2])
        image_b_std = np.std(image[:, :, 2])

        metadata.append({
            'id': image_id,
            'organ': organ,
            'data_source': 'MoNuSeg',
            'stain': 'H&E',
            'image_height': image.shape[0],
            'image_width': image.shape[1],
            'pixel_size': pixel_size,
            'tissue_thickness': np.nan,
            'rle': np.nan,
            'age': np.nan,
            'sex': np.nan,
            'image_r_mean': image_r_mean,
            'image_r_std': image_r_std,
            'image_g_mean': image_g_mean,
            'image_g_std': image_g_std,
            'image_b_mean': image_b_mean,
            'image_b_std': image_b_std,
            'mask_area': np.nan,
            'image_filename': image_filename,
            'mask_filename': np.nan
        })

    df_metadata = pd.DataFrame(metadata)
    df_metadata = df_metadata.loc[df_metadata['organ'].isin(['kidney', 'prostate', 'colon', 'lung']), :].reset_index(drop=True)
    df_metadata.to_csv(dataset_path / 'metadata.csv', index=False)
    logging.info(f'Saved metadata.csv to {dataset_path}')
    logging.info(f'Dataset Shape: {df_metadata.shape} - Memory Usage: {df_metadata.memory_usage().sum() / 1024 ** 2:.2f} MB')
