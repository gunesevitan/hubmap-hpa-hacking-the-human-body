import numpy as np
import pandas as pd


def _get_folds(df, df_folds):

    """
    Merge training set with pre-computed folds

    Parameters
    ----------
    df (pandas.DataFrame of shape (n_rows, n_columns)): Training dataframe
    df_folds (pandas.DataFrame of shape (n_rows, n_folds)): Folds

    Returns
    -------
    df (pandas.DataFrame of shape (n_rows, n_columns + n_folds)): Training data with folds
    """

    df = df.merge(df_folds, on='id', how='left')

    return df


def _get_hubmap_kidney_segmentation_data(df, df_hubmap_kidney_segmentation, mask_area_range=(5000, 500000)):

    """
    Merge training set with HuBMAP Kidney Segmentation data

    Parameters
    ----------
    df (pandas.DataFrame of shape (n_rows, n_columns)): Training dataframe
    df_hubmap_kidney_segmentation (pandas.DataFrame of shape (n_rows, n_columns)): HuBMAP Kidney Segmentation data

    Returns
    -------
    df (pandas.DataFrame of shape (n_rows, n_columns)): Training data with HuBMAP Kidney Segmentation data
    """

    df_hubmap_kidney_segmentation = df_hubmap_kidney_segmentation.loc[df_hubmap_kidney_segmentation['mask_area'].notna(), :]
    df_hubmap_kidney_segmentation = df_hubmap_kidney_segmentation.loc[
        (df_hubmap_kidney_segmentation['mask_area'] >= mask_area_range[0]) & (df_hubmap_kidney_segmentation['mask_area'] <= mask_area_range[1]), :
    ].reset_index(drop=True)

    df = pd.concat((
        df, df_hubmap_kidney_segmentation
    ), axis=0, ignore_index=True)
    df[[column for column in df.columns if column.startswith('fold')]] = df[[column for column in df.columns if column.startswith('fold')]].fillna(0).astype(np.uint8)

    return df


def preprocess_datasets(
        df_train, df_test, df_folds, df_hubmap_kidney_segmentation,
        hubmap_kidney_segmentation_sample_count, mask_area_range=(5000, 500000)
):

    """
    Preprocess training and test sets

    Parameters
    ----------
    df_train (pandas.DataFrame of shape (n_rows, n_columns)): Training dataframe
    df_test (pandas.DataFrame of shape (n_rows, n_columns)): Test dataframe
    df_folds (pandas.DataFrame of shape (n_rows, n_folds)): Folds
    hubmap_kidney_segmentation_sample_count (int): Number of samples from HuBMAP Kidney Segmentation data
    df_hubmap_kidney_segmentation (pandas.DataFrame of shape (n_rows, n_columns)): HuBMAP Kidney Segmentation dataframe
    mask_area_range (tuple): Lower and upper bounds of mask area thresholds for HuBMAP Kidney Segmentation masks

    Returns
    -------
    df_train (pandas.DataFrame of shape (n_rows, n_columns)): Training dataframe
    df_test (pandas.DataFrame of shape (n_rows, n_columns)): Test dataframe
    """

    df_train = _get_folds(df=df_train, df_folds=df_folds)
    if hubmap_kidney_segmentation_sample_count > 0:
        df_train = _get_hubmap_kidney_segmentation_data(
            df=df_train,
            df_hubmap_kidney_segmentation=df_hubmap_kidney_segmentation.sample(n=hubmap_kidney_segmentation_sample_count),
            mask_area_range=mask_area_range
        )

    return df_train, df_test
