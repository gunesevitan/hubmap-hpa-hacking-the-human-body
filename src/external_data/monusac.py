import logging
import sys
from glob import glob
from tqdm import tqdm
from shutil import copy2
import xml.etree.ElementTree as ET
import numpy as np
import pandas as pd
import tifffile

sys.path.append('..')
import settings


organs = {
    'TCGA-55-1594': 'lung',
    'TCGA-69-7760': 'lung',
    'TCGA-69-A59K': 'lung',
    'TCGA-73-4668': 'lung',
    'TCGA-78-7220': 'lung',
    'TCGA-86-7713': 'lung',
    'TCGA-86-8672': 'lung',
    'TCGA-L4-A4E5': 'lung',
    'TCGA-MP-A4SY': 'lung',
    'TCGA-49-6743': 'lung',
    'TCGA-50-6591': 'lung',
    'TCGA-55-7570': 'lung',
    'TCGA-55-7573': 'lung',
    'TCGA-73-4662': 'lung',
    'TCGA-78-7152': 'lung',
    'TCGA-MP-A4T7': 'lung',
    'TCGA-5P-A9K0': 'kidney',
    'TCGA-B9-A44B': 'kidney',
    'TCGA-B9-A8YI': 'kidney',
    'TCGA-DW-7841': 'kidney',
    'TCGA-EV-5903': 'kidney',
    'TCGA-F9-A97G': 'kidney',
    'TCGA-G7-A8LD': 'kidney',
    'TCGA-MH-A560': 'kidney',
    'TCGA-P4-AAVK': 'kidney',
    'TCGA-SX-A7SR': 'kidney',
    'TCGA-UZ-A9PO': 'kidney',
    'TCGA-UZ-A9PU': 'kidney',
    'TCGA-2Z-A9JG': 'kidney',
    'TCGA-2Z-A9JN': 'kidney',
    'TCGA-DW-7838': 'kidney',
    'TCGA-DW-7963': 'kidney',
    'TCGA-F9-A8NY': 'kidney',
    'TCGA-IZ-A6M9': 'kidney',
    'TCGA-MH-A55W': 'kidney',
    'TCGA-A2-A0CV': 'breast',
    'TCGA-A2-A0ES': 'breast',
    'TCGA-B6-A0WZ': 'breast',
    'TCGA-BH-A18T': 'breast',
    'TCGA-D8-A1X5': 'breast',
    'TCGA-E2-A154': 'breast',
    'TCGA-E9-A22B': 'breast',
    'TCGA-E9-A22G': 'breast',
    'TCGA-EW-A6SD': 'breast',
    'TCGA-S3-AA11': 'breast',
    'TCGA-A2-A04X': 'breast',
    'TCGA-D8-A3Z6': 'breast',
    'TCGA-E2-A108': 'breast',
    'TCGA-EW-A6SB': 'breast',
    'TCGA-EJ-5495': 'prostate',
    'TCGA-EJ-5505': 'prostate',
    'TCGA-EJ-5517': 'prostate',
    'TCGA-G9-6342': 'prostate',
    'TCGA-G9-6499': 'prostate',
    'TCGA-J4-A67Q': 'prostate',
    'TCGA-J4-A67T': 'prostate',
    'TCGA-KK-A59X': 'prostate',
    'TCGA-KK-A6E0': 'prostate',
    'TCGA-KK-A7AW': 'prostate',
    'TCGA-V1-A8WL': 'prostate',
    'TCGA-V1-A9O9': 'prostate',
    'TCGA-X4-A8KQ': 'prostate',
    'TCGA-YL-A9WY': 'prostate',
    'TCGA-G9-6356': 'prostate',
    'TCGA-G9-6367': 'prostate',
    'TCGA-VP-A87E': 'prostate',
    'TCGA-VP-A87H': 'prostate',
    'TCGA-X4-A8KS': 'prostate',
    'TCGA-YL-A9WL': 'prostate',
}


if __name__ == '__main__':

    dataset_path = settings.DATA / 'external_data' / 'MoNuSAC'
    (dataset_path / 'images').mkdir(parents=True, exist_ok=True)
    patient_directories = glob(str(dataset_path / 'TCGA*'))

    metadata = []

    for patient_directory in tqdm(patient_directories):

        patient_id = '-'.join(patient_directory.split('/')[-1].split('-')[:3])

        image_filenames = glob(f'{patient_directory}/*.tif')
        for image_filename in image_filenames:

            image_id = image_filename.split("/")[-1].split(".")[0]
            annotations = ET.parse(f'{patient_directory}/{image_id}.xml').getroot()
            image = tifffile.imread(image_filename)

            copy2(image_filename, dataset_path / 'images')

            # Extract metadata from image
            image_r_mean = np.mean(image[:, :, 0])
            image_r_std = np.std(image[:, :, 0])
            image_g_mean = np.mean(image[:, :, 1])
            image_g_std = np.std(image[:, :, 1])
            image_b_mean = np.mean(image[:, :, 2])
            image_b_std = np.std(image[:, :, 2])

            metadata.append({
                'id': image_id,
                'organ': organs[patient_id],
                'data_source': 'MoNuSAC',
                'stain': 'H&E',
                'image_height': image.shape[0],
                'image_width': image.shape[1],
                'pixel_size': float(annotations.attrib['MicronsPerPixel']),
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
                'image_filename': str(dataset_path / 'images' / f'{image_id}.tif'),
                'mask_filename': np.nan
            })

    df_metadata = pd.DataFrame(metadata)
    df_metadata = df_metadata.loc[df_metadata['organ'] != 'breast', :]
    df_metadata.to_csv(dataset_path / 'metadata.csv', index=False)
    logging.info(f'Saved metadata.csv to {dataset_path}')
    logging.info(f'Dataset Shape: {df_metadata.shape} - Memory Usage: {df_metadata.memory_usage().sum() / 1024 ** 2:.2f} MB')
