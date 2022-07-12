import logging
import pathlib
from tqdm import tqdm
import numpy as np
import pandas as pd
import tifffile
import matplotlib.pyplot as plt
import seaborn as sns

import settings
import annotation_utils


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


def visualize_annotations(image, rle_mask, polygons, metadata, path=None):

    """
    Visualize image along with its annotations

    Parameters
    ----------
    image (path-like str or numpy.ndarray of shape (height, width)): Image path relative to root/data or image array
    rle_mask (str): Run-length encoded segmentation mask string
    polygons (list of shape (n_polygons, n_points, 2)): Polygons
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

    fig, axes = plt.subplots(figsize=(48, 16), ncols=3)

    axes[0].imshow(image)
    axes[1].imshow(image)

    if rle_mask is not None:
        mask = annotation_utils.decode_rle_mask(rle_mask=rle_mask, shape=image.shape[:2]).T
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
        f'Image ID {metadata["id"]} - {row["organ"]} - {row["data_source"]} - {int(row["age"])} - {row["sex"]}\nImage Shape: {row["img_height"]}x{row["img_width"]} - Pixel Size: {row["pixel_size"]}µm - Tissue Thickness: {row["tissue_thickness"]}µm\n{int(row["n_polygons"])} Annotations',
        fontsize=30
    )

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
    VISUALIZE_CATEGORICAL_FEATURES = True

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
                polygons=polygons,
                metadata=row.to_dict(),
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
