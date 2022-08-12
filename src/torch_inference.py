import logging
import argparse
from glob import glob
from tqdm import tqdm
import yaml
import pathlib
import json
import numpy as np
import pandas as pd
import cv2
import tifffile
import torch

import settings
import annotation_utils
import visualization
import transforms
import torch_modules
import evaluation
import test_time_augmentations


def load_model(path, folds=(1, 2, 3, 4, 5), verbose=False):

    """
    Load models and config file from the given path

    Parameters
    ----------
    path (str): Path of the model directory
    folds (tuple): Tuple of folds to load
    verbose (bool): Verbosity flag

    Returns
    -------
    config (dict): Dictionary of model configurations
    models (list): List of trained models
    """

    config = yaml.load(open(f'{path}/config.yaml', 'r'), Loader=yaml.FullLoader)
    # Set encoder weights to None, so the pretrained weights won't be downloaded
    if config['model_parameters']['model_module'] == 'smp':
        config['model_parameters']['model_args']['encoder_weights'] = None

    model_filenames = sorted(glob(f'{path}/*.pt'))
    models = {}

    for fold, model_filename in enumerate(model_filenames, 1):

        if fold not in folds:
            continue
        else:
            model = torch_modules.SemanticSegmentationModel(
                config['model_parameters']['model_module'],
                config['model_parameters']['model_class'],
                config['model_parameters']['model_args']
            )
            model.load_state_dict(torch.load(model_filename))
            model = model.to(config['training_parameters']['device'])
            model.eval()
            models[fold] = model
            logging.info(f'Loaded pretrained weights from {model_filename}')

    if verbose:
        logging.info(json.dumps(config, indent=2))

    return config, models


if __name__ == '__main__':

    save_mask = True
    visualize_predictions = False

    parser = argparse.ArgumentParser()
    parser.add_argument('dataset_path', type=str)
    parser.add_argument('model_path', type=str)
    args = parser.parse_args()

    dataset_path = pathlib.Path(args.dataset_path)
    df_metadata = pd.read_csv(dataset_path / 'metadata.csv')
    unet_semantic_segmentation_raw_config, unet_semantic_segmentation_raw_models = load_model(
        path=args.model_path,
        folds=(1, 2, 3, 4, 5),
        verbose=True
    )
    dataset_transforms = transforms.get_semantic_segmentation_transforms(**unet_semantic_segmentation_raw_config['transform_parameters'])

    for idx, row in tqdm(df_metadata.iterrows(), total=df_metadata.shape[0]):

        if row['image_filename'].split('.')[-1] == 'tiff' or row['image_filename'].split('.')[-1] == 'tif':
            image = tifffile.imread(row['image_filename'])
        else:
            image = cv2.imread(row['image_filename'])
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        image_resized = cv2.resize(
            image,
            dsize=(
                unet_semantic_segmentation_raw_config['transform_parameters']['resize_width'],
                unet_semantic_segmentation_raw_config['transform_parameters']['resize_height']
            ),
            interpolation=cv2.INTER_CUBIC
        )

        inputs = [
            image_resized,
            test_time_augmentations.horizontal_flip(image_resized),
            test_time_augmentations.vertical_flip(image_resized),
            test_time_augmentations.horizontal_flip(test_time_augmentations.vertical_flip(image_resized))
        ]
        inputs = torch.cat([
            torch.unsqueeze(dataset_transforms['test'](image=image)['image'], dim=0)
            for image in inputs
        ], dim=0)
        inputs = inputs.to('cuda')

        predictions_mask = np.zeros((
            len(inputs),
            unet_semantic_segmentation_raw_config['transform_parameters']['resize_width'],
            unet_semantic_segmentation_raw_config['transform_parameters']['resize_height']
        ), dtype=np.float32)

        for fold, model in unet_semantic_segmentation_raw_models.items():
            with torch.no_grad():
                outputs = unet_semantic_segmentation_raw_models[fold](inputs)

            fold_predictions_mask = outputs.detach().cpu()
            fold_predictions_mask = torch.sigmoid(torch.squeeze(fold_predictions_mask, dim=1)).numpy().astype(np.float32)
            predictions_mask += (fold_predictions_mask / len(unet_semantic_segmentation_raw_models))

        # Apply inverse of test-time augmentations and aggregate predictions
        predictions_mask[1, :, :] = test_time_augmentations.horizontal_flip(predictions_mask[1, :, :])
        predictions_mask[2, :, :] = test_time_augmentations.vertical_flip(predictions_mask[2, :, :])
        predictions_mask[3, :, :] = test_time_augmentations.horizontal_flip(test_time_augmentations.vertical_flip(predictions_mask[3, :, :]))
        predictions_mask = np.mean(predictions_mask, axis=0)
        predictions_mask = cv2.resize(predictions_mask, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_CUBIC)

        try:
            label_threshold = unet_semantic_segmentation_raw_config['inference_parameters']['label_thresholds'][row['data_source']][row['organ']]
        except KeyError:
            # Set label threshold to 0.1 for unseen organs or data sources
            label_threshold = 0.1

        if visualize_predictions:
            predictions_evaluation_summary = evaluation.evaluate_predictions(
                ground_truth=None,
                predictions=predictions_mask,
                threshold=label_threshold,
                thresholds=unet_semantic_segmentation_raw_config['inference_parameters']['label_threshold_range']
            )
            predictions_mask = np.uint8(predictions_mask >= label_threshold)
            visualization.visualize_predictions(
                image=image,
                ground_truth=None,
                predictions=predictions_mask,
                metadata=row.to_dict(),
                evaluation_summary=predictions_evaluation_summary
            )
        else:
            predictions_mask = np.uint8(predictions_mask >= label_threshold)

        df_metadata.loc[idx, 'rle'] = annotation_utils.encode_rle_mask(predictions_mask)
        if save_mask:
            np.save(str(dataset_path / 'prediction_masks' / f'{row["id"]}.npy'), predictions_mask)

    df_metadata.to_csv(dataset_path / 'metadata.csv', index=False)
    logging.info(f'Saved metadata.csv to {dataset_path}')
