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


def _get_external_data(df, external_data):

    """
    Merge training set external data

    Parameters
    ----------
    df (pandas.DataFrame of shape (n_rows, n_columns)): Training dataframe
    external_data (dict): Dictionary of external dataset metadata dataframes

    Returns
    -------
    df (pandas.DataFrame of shape (n_rows, n_columns)): Training data and external data
    """

    df = pd.concat([df] + list(external_data.values()), axis=0, ignore_index=True)
    df = df.loc[df['rle'].notna(), :].reset_index(drop=True)
    df.loc[df['rle_corrected'].isna(), 'rle_corrected'] = df.loc[df['rle_corrected'].isna(), 'rle'].values
    df[[column for column in df.columns if column.startswith('fold')]] = df[[column for column in df.columns if column.startswith('fold')]].fillna(0).astype(np.uint8)

    return df


def preprocess_datasets(df_train, df_test, df_folds, external_data):

    """
    Preprocess training and test sets

    Parameters
    ----------
    df_train (pandas.DataFrame of shape (n_rows, n_columns)): Training dataframe
    df_test (pandas.DataFrame of shape (n_rows, n_columns)): Test dataframe
    df_folds (pandas.DataFrame of shape (n_rows, n_folds)): Folds
    external_data (dict): Dictionary of external dataset metadata dataframes

    Returns
    -------
    df_train (pandas.DataFrame of shape (n_rows, n_columns)): Training dataframe
    df_test (pandas.DataFrame of shape (n_rows, n_columns)): Test dataframe
    """

    df_train = _get_folds(df=df_train, df_folds=df_folds)
    df_train = _get_external_data(df=df_train, external_data=external_data)

    return df_train, df_test
