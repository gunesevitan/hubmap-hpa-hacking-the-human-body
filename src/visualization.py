import logging
import pathlib
from tqdm import tqdm
import numpy as np
import pandas as pd
import cv2
import tifffile
import matplotlib.pyplot as plt
import seaborn as sns
import albumentations as A

import settings
import annotation_utils
import preprocessing


def visualize_categorical_feature_distribution(df, categorical_feature, path=None):

    """
    Visualize distribution of given categorical feature in given dataframe

    Parameters
    ----------
    df (pandas.DataFrame of shape (n_rows, n_columns)): Dataframe with given categorical feature column
    categorical_feature (str): Name of the categorical feature column
    path (path-like str or None): Path of the output file or None (if path is None, plot is displayed with selected backend)
    """

    fig, ax = plt.subplots(figsize=(24, df[categorical_feature].value_counts().shape[0]), dpi=100)
    sns.barplot(
        x=df[categorical_feature].value_counts().values,
        y=df[categorical_feature].value_counts().index,
        color='tab:blue',
        ax=ax
    )
    ax.set_xlabel('')
    ax.set_ylabel('')
    ax.set_yticklabels([
        f'{x} ({value_count:,})' for value_count, x in zip(
            df[categorical_feature].value_counts().values,
            df[categorical_feature].value_counts().index
        )
    ])
    ax.tick_params(axis='x', labelsize=15, pad=10)
    ax.tick_params(axis='y', labelsize=15, pad=10)
    ax.set_title(f'Value Counts {categorical_feature}', size=20, pad=15)

    if path is None:
        plt.show()
    else:
        plt.savefig(path)
        plt.close(fig)


def visualize_annotations(image, rle_mask, polygons, metadata, crop_black_border=False, crop_background=False, path=None):

    """
    Visualize image along with its annotations

    Parameters
    ----------
    image (path-like str or numpy.ndarray of shape (height, width, 3)): Image path relative to root/data or image array
    rle_mask (str): Run-length encoded segmentation mask string
    polygons (list of shape (n_polygons, n_points, 2)): Polygons
    metadata (dict): Dictionary of metadata used in the visualization title
    crop_black_border (bool): Whether to crop black border or not
    crop_background (bool): Whether to crop background or not
    path (path-like str or None): Path of the output file or None (if path is None, plot is displayed with selected backend)
    """

    if isinstance(image, pathlib.Path) or isinstance(image, str):
        # Read image from the given path
        image_path = image
        image = tifffile.imread(str(settings.DATA / image_path))

    elif isinstance(image, np.ndarray):
        title = ''

    else:
        # Raise TypeError if image argument is not an array-like object or a path-like string
        raise TypeError('Image is not an array or path.')

    if rle_mask is not None:
        mask = annotation_utils.decode_rle_mask(rle_mask=rle_mask, shape=image.shape[:2])
        if metadata['data_source'] == 'HPA' or metadata['data_source'] == 'Hubmap':
            mask = mask.T

    image, mask = preprocessing.crop_image(
        image=image,
        mask=mask,
        crop_black_border=crop_black_border,
        crop_background=crop_background
    )

    fig, axes = plt.subplots(figsize=(48, 20), ncols=3)

    axes[0].imshow(image)
    axes[1].imshow(image)

    if rle_mask is not None:
        axes[1].imshow(mask, alpha=0.5)
        axes[2].imshow(mask)

    if polygons is not None:
        for polygon in polygons:
            polygon = np.array(polygon)
            axes[1].plot(polygon[:, 0], polygon[:, 1], linewidth=2, color='red', alpha=0.5)
            axes[2].plot(polygon[:, 0], polygon[:, 1], linewidth=2, color='red')

    for i in range(3):
        axes[i].set_xlabel('')
        axes[i].set_ylabel('')
        axes[i].tick_params(axis='x', labelsize=15, pad=10)
        axes[i].tick_params(axis='y', labelsize=15, pad=10)

    axes[0].set_title('Image', size=20, pad=15)
    axes[1].set_title('Image + Mask and Polygons', size=20, pad=15)
    axes[2].set_title('Mask and Polygons', size=20, pad=15)

    fig.suptitle(
        f'''
        Image ID {metadata["id"]} - {metadata["organ"]} - {metadata["data_source"]} - {metadata["age"]} - {metadata["sex"]}
        Image Shape: {metadata["image_height"]}x{metadata["image_width"]} - Pixel Size: {metadata["pixel_size"]}µm - Tissue Thickness: {metadata["tissue_thickness"]}µm
        {metadata["n_polygons"]} Annotations'
        ''',
        fontsize=25
    )

    if path is None:
        plt.show()
    else:
        plt.savefig(path)
        plt.close(fig)


def visualize_predictions(image, ground_truth, predictions, metadata, evaluation_summary, path=None):

    """
    Visualize image along with its annotations and predictions

    Parameters
    ----------
    image (numpy.ndarray of shape (height, width, 3)): Image array
    ground_truth (numpy.ndarray of shape (height, width)): Ground-truth mask array
    predictions (numpy.ndarray of shape (height, width)): Predictions mask array
    metadata (dict): Dictionary of metadata used in the visualization title
    evaluation_summary (dict): Dictionary of evaluation summary used in the visualization title
    path (path-like str or None): Path of the output file or None (if path is None, plot is displayed with selected backend)
    """

    if ground_truth is not None:
        dice_coefficient = round(np.median(list(evaluation_summary['scores']['dice_coefficients'].values())), 4)
        intersection_over_union = round(np.median(list(evaluation_summary['scores']['intersection_over_unions'].values())), 4)
        ground_truth_evaluation = f'Mean: {evaluation_summary["statistics"]["ground_truth"]["mean"]:4f} - Sum: {evaluation_summary["statistics"]["ground_truth"]["sum"]} - Object Count: {evaluation_summary["spatial_properties"]["ground_truth"]["n_objects"]}'
    else:
        dice_coefficient = ''
        intersection_over_union = ''
        ground_truth_evaluation = ''

    predictions_evaluation = f'Mean: {evaluation_summary["statistics"]["predictions"]["mean"]:4f} - Sum: {evaluation_summary["statistics"]["predictions"]["sum"]} - Object Count: {evaluation_summary["spatial_properties"]["predictions"]["n_objects"]}'

    if isinstance(image, np.ndarray) is False:
        # Raise TypeError if image argument is not an array-like object
        raise TypeError('Image is not an array')

    fig, axes = plt.subplots(figsize=(32, 20), ncols=2)

    axes[0].imshow(image)
    if ground_truth is not None:
        axes[0].imshow(ground_truth, alpha=0.5)
    axes[1].imshow(image)
    axes[1].imshow(predictions, alpha=0.5)

    for i in range(2):
        axes[i].set_xlabel('')
        axes[i].set_ylabel('')
        axes[i].tick_params(axis='x', labelsize=15, pad=10)
        axes[i].tick_params(axis='y', labelsize=15, pad=10)

    axes[0].set_title('Image + Ground-truth\n' + ground_truth_evaluation, size=25, pad=15)
    axes[1].set_title('Image + Predictions\n' + predictions_evaluation, size=25, pad=15)
    fig.suptitle(
        f'''
        Image ID {metadata["id"]} - {metadata["organ"]} - {metadata["data_source"]} - {int(metadata["age"])} - {metadata["sex"]}
        Image Shape: {metadata["image_height"]}x{metadata["image_width"]} - Pixel Size: {metadata["pixel_size"]}µm - Tissue Thickness: {metadata["tissue_thickness"]}µm
        Dice Coefficient: {dice_coefficient} - Intersection over Union: {intersection_over_union}
        ''',
        fontsize=30
    )

    if path is None:
        plt.show()
    else:
        plt.savefig(path)
        plt.close(fig)


def visualize_transforms(image, mask, transforms, path=None):

    """
    Visualize image along with its annotations and predictions

    Parameters
    ----------
    image (numpy.ndarray of shape (height, width, 3)): Image array
    mask (numpy.ndarray of shape (height, width)): Mask array
    transforms (albumentations.Compose): Transforms to apply on image and mask if it is given
    path (path-like str or None): Path of the output file or None (if path is None, plot is displayed with selected backend)
    """

    fig, axes = plt.subplots(figsize=(32, 20), ncols=2)
    axes[0].imshow(image)

    if mask is not None:
        rgb_mask = np.moveaxis(np.stack([mask] * 3), 0, -1) * 255
        axes[0].imshow(rgb_mask, alpha=0.5)
        # Apply transforms to image and mask
        transformed = transforms(image=image, mask=mask)
        transformed_image = transformed['image']
        transformed_mask = transformed['mask']
        transformed_rgb_mask = np.moveaxis(np.stack([transformed_mask] * 3), 0, -1) * 255
        axes[1].imshow(transformed_image)
        axes[1].imshow(transformed_rgb_mask, alpha=0.5)
    else:
        # Apply transforms to image
        transformed = transforms(image=image)
        transformed_image = transformed['image']
        axes[1].imshow(transformed_image)

    for i in range(2):
        axes[i].set_xlabel('')
        axes[i].set_ylabel('')
        axes[i].tick_params(axis='x', labelsize=15, pad=10)
        axes[i].tick_params(axis='y', labelsize=15, pad=10)

    axes[0].set_title(f'{image.shape} - Mean: {image.mean():.2f} - Std: {image.std():.2f}\nMin: {image.min():.2f} - Max: {image.max():.2f}', size=20, pad=15)
    axes[1].set_title(f'{transformed_image.shape} - Mean: {transformed_image.mean():.2f} - Std: {transformed_image.std():.2f}\nMin: {transformed_image.min():.2f} - Max: {transformed_image.max():.2f}', size=20, pad=15)

    if path is None:
        plt.show()
    else:
        plt.savefig(path)
        plt.close(fig)


def visualize_learning_curve(training_losses, validation_losses, path=None):

    """
    Visualize learning curves of the models

    Parameters
    ----------
    training_losses (array-like of shape (n_epochs or n_steps)): Array of training losses computed after every epoch or step
    validation_losses (array-like of shape (n_epochs or n_steps)): Array of validation losses computed after every epoch or step
    path (str or None): Path of the output file (if path is None, plot is displayed with selected backend)
    """

    fig, ax = plt.subplots(figsize=(32, 8), dpi=100)
    sns.lineplot(
        x=np.arange(1, len(training_losses) + 1),
        y=training_losses,
        ax=ax,
        label='train_loss'
    )
    if validation_losses is not None:
        sns.lineplot(
            x=np.arange(1, len(validation_losses) + 1),
            y=validation_losses,
            ax=ax,
            label='val_loss'
        )
    ax.set_xlabel('Epochs/Steps', size=15, labelpad=12.5)
    ax.set_ylabel('Loss', size=15, labelpad=12.5)
    ax.tick_params(axis='x', labelsize=12.5, pad=10)
    ax.tick_params(axis='y', labelsize=12.5, pad=10)
    ax.legend(prop={'size': 18})
    ax.set_title('Learning Curve', size=20, pad=15)

    if path is None:
        plt.show()
    else:
        plt.savefig(path)
        plt.close(fig)


def visualize_scores(df_scores, path=None):

    """
    Visualize metric scores of multiple models with error bars

    Parameters
    ----------
    df_scores (pandas.DataFrame of shape (n_folds, 6)): DataFrame of metric scores
    path (str or None): Path of the output file (if path is None, plot is displayed with selected backend)
    """

    # Create mean and std of scores for error bars
    df_scores = df_scores.T
    column_names = df_scores.columns.to_list()
    df_scores['mean'] = df_scores[column_names].mean(axis=1)
    df_scores['std'] = df_scores[column_names].std(axis=1)

    fig, ax = plt.subplots(figsize=(24, 8))
    ax.barh(
        y=np.arange(df_scores.shape[0]),
        width=df_scores['mean'],
        xerr=df_scores['std'],
        align='center',
        ecolor='black',
        capsize=10
    )
    ax.set_yticks(np.arange(df_scores.shape[0]))
    ax.set_yticklabels([
        f'{metric}\n{mean:.4f} (±{std:.4f})' for metric, mean, std in zip(
            df_scores.index,
            df_scores['mean'].values,
            df_scores['std'].values
        )
    ])
    ax.set_xlabel('')
    ax.tick_params(axis='x', labelsize=15, pad=10)
    ax.tick_params(axis='y', labelsize=15, pad=10)
    ax.set_title('Metric Scores', size=20, pad=15)

    if path is None:
        plt.show()
    else:
        plt.savefig(path)
        plt.close(fig)


if __name__ == '__main__':

    df_train = pd.read_csv(settings.DATA / 'train_metadata.csv')

    VISUALIZE_IMAGES = False
    VISUALIZE_CATEGORICAL_FEATURES = False
    VISUALIZE_TRANSFORMS = False

    if VISUALIZE_IMAGES:

        annotations_visualizations_directory = settings.EDA / 'annotations'
        annotations_visualizations_directory.mkdir(parents=True, exist_ok=True)

        for idx, row in tqdm(df_train.iterrows(), total=df_train.shape[0]):

            image = tifffile.imread(str(settings.DATA / 'train_images' / f'{row["id"]}.tiff'))
            rle_mask = df_train.loc[df_train['id'] == row['id'], 'rle'].values[0]
            polygons = annotation_utils.get_segmentation_polygons_from_json(
                annotation_directory='train_annotations',
                image_id=row['id']
            )

            visualize_annotations(
                image=image,
                rle_mask=rle_mask,
                polygons=None,
                metadata=row.to_dict(),
                crop_black_border=True,
                crop_background=True,
                path=annotations_visualizations_directory / f'{row["id"]}_annotations.png'
            )

        logging.info(f'Saved annotations visualizations to {annotations_visualizations_directory}')

    if VISUALIZE_CATEGORICAL_FEATURES:

        df_train['image_dimensions'] = df_train['img_height'].astype(str) + 'x' + df_train['img_width'].astype(str)

        categorical_features = ['organ', 'image_dimensions']
        for categorical_feature in categorical_features:
            visualize_categorical_feature_distribution(
                df=df_train,
                categorical_feature=categorical_feature,
                path=settings.EDA / f'{categorical_feature}_distribution.png'
            )

        logging.info(f'Saved categorical feature distribution visualizations to {settings.EDA}')

    if VISUALIZE_TRANSFORMS:

        image_idx = 0
        image = tifffile.imread(df_train.loc[image_idx, 'image_filename'])
        mask = annotation_utils.decode_rle_mask(df_train.loc[image_idx, 'rle'], shape=image.shape[:2]).T

        transforms = A.Compose([
            A.RandomResizedCrop(
                height=512,
                width=512,
                scale=(0.5, 0.5),
                ratio=(0.5, 1.5),
                interpolation=cv2.INTER_NEAREST,
                always_apply=True
            ),
            A.HorizontalFlip(p=0.0),
            A.VerticalFlip(p=0.0),
            A.RandomRotate90(p=0.0),
            A.ShiftScaleRotate(
                shift_limit=0.1,
                scale_limit=0.25,
                rotate_limit=180,
                interpolation=cv2.INTER_NEAREST,
                border_mode=cv2.BORDER_CONSTANT,
                value=(245, 245, 245),
                mask_value=(0, 0, 0),
                p=0.0
            ),
            A.RGBShift(
                r_shift_limit=20,
                g_shift_limit=20,
                b_shift_limit=20,
                p=0.0
            )

        ])
        visualize_transforms(
            image=image,
            mask=mask,
            transforms=transforms
        )

    df_hubmap_kidney_segmentation_metadata = pd.read_csv(settings.DATA / 'hubmap_kidney_segmentation_metadata.csv')
    df_hubmap_kidney_segmentation_metadata = df_hubmap_kidney_segmentation_metadata.loc[df_hubmap_kidney_segmentation_metadata['mask_area'] > 5000].reset_index(drop=True)


    print(df_hubmap_kidney_segmentation_metadata.shape, df_hubmap_kidney_segmentation_metadata['mask_area'].mean(), df_hubmap_kidney_segmentation_metadata['mask_area'].min(), df_hubmap_kidney_segmentation_metadata['mask_area'].max())
    idx = np.argmin(df_hubmap_kidney_segmentation_metadata['mask_area'].values)
    image = cv2.imread(df_hubmap_kidney_segmentation_metadata.loc[idx, 'image_filename'])
    mask = cv2.imread(df_hubmap_kidney_segmentation_metadata.loc[idx, 'mask_filename'], -1)


    plt.imshow(image)
    plt.imshow(mask, alpha=0.5)
    plt.show()

    idx = np.argmax(df_hubmap_kidney_segmentation_metadata['mask_area'].values)
    image = cv2.imread(df_hubmap_kidney_segmentation_metadata.loc[idx, 'image_filename'], -1)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    mask = cv2.imread(df_hubmap_kidney_segmentation_metadata.loc[idx, 'mask_filename'], -1)

    plt.imshow(image)
    #plt.imshow(mask, alpha=0.5)
    plt.show()