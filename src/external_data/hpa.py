import logging
import sys
import time
from glob import glob
from tqdm import tqdm
from itertools import product
from io import BytesIO
from PIL import Image
import numpy as np
import pandas as pd
import cv2
import requests

sys.path.append('..')
import settings
import annotation_utils


if __name__ == '__main__':

    dataset_path = settings.DATA / 'external_data' / 'HPA'
    download_images = False
    base_url = 'https://images.proteinatlas.org/dictionary_images'
    filenames = {
        'cerebral_cortex': 'fileup5ebbc37a92939942766250_files',
        'hippocampus': 'fileup5ebbca2a7ae8f740345048_files',
        'caudate': 'fileup5ebbc5e1b5e70890321351_files',
        'cerebellum': 'fileup5ebbb3dce6022592461088_files',
        'thyroid_gland': 'fileup5ebbf958005df256832693_files',
        'parathyroid_gland': 'fileup5ebbf0d6cfd5d883024748_files',
        'adrenal_gland': 'fileup5f06d46f44e8d572433101_files',
        'nasopharynx': 'fileup5f06f9447fdc3040954725_files',
        'bronchus': 'fileup5f4e01222bc9d568079076_files',
        'lung': 'fileup5e998cc3050ef333190901_files',
        'oral_mucosa': 'fileup5f466d2362fed039882135_files',
        'salivary_gland': 'fileup5e999702d6c85081739928_files',
        'esophagus': 'fileup5e9ead0354d20610399668_files',
        'stomach': 'fileup5e999bb7de2ed268599831_files',
        'duodenum': 'fileup5eb53a11aaaa3012578042_files',
        'small_intestine': 'fileup5e999d30cda49319915832_files',
        'colon': 'fileup5e9f0102597ae655481068_files',
        'rectum': 'fileup5eb553602a1f1421695742_files',
        'liver': 'fileup5eb56adb412e1847438951_files',
        'gallbladder': 'fileup5eb562531fd5b014875186_files',
        'pancreas': 'fileup5eb986535b8a5956243229_files',
        'kidney': 'fileup5f9fe9fbd8ca4499343163_files',
        'urinary_bladder': 'fileup5ebaa17e379a7989345256_files',
        'testis': 'fileup5f06fc18edb78237925031_files',
        'epididymis': 'fileup5ebd2c7669849581627371_files',
        'seminal_vesicle': 'fileup5ebd3789c535d473950068_files',
        'prostate': 'fileup5f9bf9680f074282451440_files',
        'vagina': 'fileup5ebd210750965614708421_files',
        'ovary': 'fileup5f06f2bcd55af304062036_files',
        'fallopian_tube': 'fileup5e8614ce9ca9f213804177_files',
        'endometrium': 'fileup5ebc0d76eefaf862356006_files',
        'cervix': 'fileup5ebc07f45f9b6540379471_files',
        'placenta': 'fileup5ebd1c068ccd1074493098_files',
        'breast': 'fileup5ebc016066dfd241198334_files',
        'heart_muscle': 'fileup5ebd3d3e061e0903596864_files',
        'smooth_muscle': 'fileup5ebd4c14810b1078563387_files',
        'skeletal_muscle': 'fileup5ebd42aa4fe4e044699853_files',
        'adipose_tissue': 'fileup5ebaba76566a5401778440_files',
        'skin': 'fileup5ecd108688b6c993471583_files',
        'bone_marrow': 'fileup5ebabcd1cb253234900562_files',
        'lymph_node': 'fileup5ebac1c9f212b414431794_files',
        'tonsil': 'fileup5ebac30fe81ae808185437_files',
        'spleen': 'fileup5ebacb77d891a050777989_files',
        'appendix': 'fileup5f06e809b56ca740045170_files',
        'glioma1': 'fileup5fa2b628c50ea198951079_files',
        'glioma2': 'fileup5fa2c079afe5b288276460_files',
        'glioma3': 'fileup5fa2c3ca9d80e697641932_files',
        'head_and_neck_squamos': 'fileup5f0d9ef295c7d714027225_files',
        'head_and_neck_adenocarcinoma': 'fileup5eff25916b536518295606_files',
        'thyroid_papillary': 'fileup5f05d6c7abdb8729955127_files',
        'thyroid_follicular': 'fileup5f05db74278cf226900731_files',
        'net_lung': 'fileup5ebeb12df2afb109142735_files',
        'net_mid_gut': 'fileup5ebeb4eb599f2268989446_files',
        'lung_adenocarcinoma1': 'fileup5f4f79a759ccf555623827_files',
        'lung_adenocarcinoma2': 'fileup5f4f7f1a363b2232903110_files',
        'lung_adenocarcinoma3': 'fileup5f4f82a481810477146910_files',
        'lung_squamous_cell_carcinoma': 'fileup5f4f890b7bd46691890386_files',
        'lung_small_cell_carcinoma': 'fileup5f4f9244a07be223276722_files',
        'stomach_adenocarcinoma1': 'fileup5f0da84f58fcd834955204_files',
        'stomach_adenocarcinoma2': 'fileup5f0db1bdcfec3327226717_files',
        'colon_adenocarcinoma1': 'fileup5fa11a2b913d7968982130_files',
        'colon_adenocarcinoma2': 'fileup5efef438d3d24848301767_files',
        'liver_hepatocellular_carcinoma1': 'fileup5f0dc18d6442d293895528_files',
        'liver_hepatocellular_carcinoma2': 'fileup5f0ef0acab00d675742680_files',
        'pancreas_adenocarcinoma1': 'fileup5fa27e0f1a614542240964_files',
        'pancreas_adenocarcinoma2': 'fileup5fa28a5f21107546252512_files',
        'kidney_renal_cell1': 'fileup5f059c86eb59c073033917_files',
        'kidney_renal_cell2': 'fileup5f059c86eb59c073033917_files',
        'kidney_renal_cell3': 'fileup5f05a322deae3968302192_files',
        'urinary_bladder_urothelial_carcinoma1': 'fileup5f05e12c93bf7104475256_files',
        'urinary_bladder_urothelial_carcinoma2': 'fileup5f50a20bc07d8503825991_files',
        'testis_seminoma': 'fileup5f05b90a39627099111821_files',
        'testis_mixed_carcinoma1': 'fileup5f05bd193b2f9136514150_files',
        'testis_mixed_carcinoma2': 'fileup5f05cc7d4ff95276930683_files',
        'testis_mixed_carcinoma3': 'fileup5f0dcdc017e9f671856970_files',
        'prostate_adenocarcinoma1': 'fileup5f50af2519b5b674181495_files',
        'prostate_adenocarcinoma2': 'fileup5f04874e91469732689313_files',
        'prostate_adenocarcinoma3': 'fileup5f50a90e6b766714243426_files',
        'ovary_serous': 'fileup5f046ae640927114691141_files',
        'ovary_mucinous': 'fileup5f0471c1d68f7245826063_files',
        'ovary_endometroid': 'fileup5f047996146f5747973304_files',
        'ovary_clear_cell': 'fileup5f047b0356dcd688814049_files',
        'endometrium_endometrioid': 'fileup5efef8d2050ac989443667_files',
        'endometrium_endometrial': 'fileup5f0ef6b456dab125204103_files',
        'cervix_squamous_cell_carcinoma': 'fileup5efeee5ab56a5938529736_files',
        'cervix_adenocarcinoma': 'fileup5f0eb9e73d928478124067_files',
        'breast_ductal_carcinoma1': 'fileup5ebea16d87f87094243401_files',
        'breast_ductal_carcinoma2': 'fileup5ebea347afce1564409062_files',
        'breast_ductal_carcinoma3': 'fileup5ebeab359a4ba424534721_files',
        'breast_lobular_carcinoma': 'fileup5ebead72d4b27132338303_files',
        'skin_melanoma1': 'fileup5f045494a01cc270171462_files',
        'skin_melanoma2': 'fileup5f045560418a3699733481_files',
        'skin_melanoma3': 'fileup5f0464cc1ed4f625346699_files',
        'skin_basal_cell_carcinoma1': 'fileup5f05a4c52716a815062332_files',
        'skin_basal_cell_carcinoma2': 'fileup5f05a562258b6438509669_files',
        'skin_squamous_cell_carcinoma': 'fileup5f05a8b1517e8121166370_files',
        'lymph_node_lymphoma1': 'fileup5f50d9b80b5a9539404777_files',
        'lymph_node_lymphoma2': 'fileup5f50efeb721a6773959676_files',
        'lymph_node_lymphoma3': 'fileup5f50f7743bb7c479450281_files',
        'lymph_node_lymphoma4': 'fileup5f50fb628ef1a883485041_files',
        'lymph_node_lymphoma5': 'fileup5f50be9391d62526773984_files'
    }
    magnification_level = 11

    if download_images:
        for organ, filename in filenames.items():

            logging.info(f'Downloading Organ: {organ} Filename: {filename} Magnification Level: {magnification_level}')

            n_rows = 1
            image_found = True
            image_tiles = {}

            while image_found:

                image_names = (
                    f'{comb[0]}_{comb[1]}.jpg'
                    for comb in list(set(product(range(n_rows), repeat=2)) - set(product(range(n_rows - 1), repeat=2)))
                )

                n_images_found = 0

                for image_name in image_names:
                    image_url = f'{base_url}/{filename}/{magnification_level}/{image_name}'
                    response = requests.get(image_url, timeout=1)
                    time.sleep(0.1)
                    if response.status_code == 200:
                        image = np.array(Image.open(BytesIO(response.content)))
                        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                        image_tiles[image_name.split('.')[0]] = np.array(image)
                        n_images_found += 1

                image_found = n_images_found > 0
                if image_found:
                    n_rows += 1

            image_tile_dimension = int(np.sqrt(len(image_tiles)))
            stitched_image = np.hstack([np.vstack([image_tiles[f'{column}_{row}'] for row in range(image_tile_dimension)]) for column in range(image_tile_dimension)])
            cv2.imwrite(str(dataset_path / 'images' / f'{organ}_he.png'), stitched_image)

    image_filenames = sorted(glob(str(dataset_path / 'images' / '*.png')))

    metadata = []

    for image_filename in tqdm(image_filenames):

        image_id = image_filename.split("/")[-1].split(".")[0]
        mask_filename = str(dataset_path / 'masks' / f'{image_id}.npy')

        image = cv2.imread(image_filename)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Extract metadata from image
        image_r_mean = np.mean(image[:, :, 0])
        image_r_std = np.std(image[:, :, 0])
        image_g_mean = np.mean(image[:, :, 1])
        image_g_std = np.std(image[:, :, 1])
        image_b_mean = np.mean(image[:, :, 2])
        image_b_std = np.std(image[:, :, 2])

        try:
            mask = np.load(mask_filename)
            # Extract metadata from mask
            mask_area = np.sum(mask)
        except FileNotFoundError:
            mask = None
            mask_area = None

        metadata.append({
            'id': image_id,
            'organ': '_'.join(image_id.split('_')[:-1]),
            'data_source': 'HPA_Dictionary',
            'stain': 'H&E',
            'image_height': image.shape[0],
            'image_width': image.shape[1],
            'pixel_size': np.nan,
            'tissue_thickness': np.nan,
            'rle': annotation_utils.encode_rle_mask(mask) if mask is not None else np.nan,
            'age': np.nan,
            'sex': np.nan,
            'image_r_mean': image_r_mean,
            'image_r_std': image_r_std,
            'image_g_mean': image_g_mean,
            'image_g_std': image_g_std,
            'image_b_mean': image_b_mean,
            'image_b_std': image_b_std,
            'mask_area': mask_area if mask_area is not None else np.nan,
            'image_filename': image_filename,
            'mask_filename': np.nan
        })

    df_metadata = pd.DataFrame(metadata)
    df_metadata.to_csv(dataset_path / 'metadata.csv', index=False)
    logging.info(f'Saved metadata.csv to {dataset_path}')
    logging.info(f'Dataset Shape: {df_metadata.shape} - Memory Usage: {df_metadata.memory_usage().sum() / 1024 ** 2:.2f} MB')
