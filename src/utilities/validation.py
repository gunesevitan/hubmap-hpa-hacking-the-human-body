import logging
import sys
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

sys.path.append('..')
import settings


def get_folds(df, n_splits, shuffle=True, random_state=42, verbose=True):

    """
    Create columns of folds

    Parameters
    ----------
    df (pandas.DataFrame of shape (n_rows, n_columns)): DataFrame with organ column
    n_splits (int): Number of folds (2 <= n_splits)
    shuffle (bool): Whether to shuffle before split or not
    random_state (int): Random seed for reproducible results
    verbose (bool): Flag for verbosity

    Returns
    -------
    df (pandas.DataFrame of shape (n_rows, n_columns)): DataFrame with segmentation, class, case and fold columns
    """

    skf = StratifiedKFold(n_splits=n_splits, shuffle=shuffle, random_state=random_state)
    for fold, (training_idx, validation_idx) in enumerate(skf.split(X=df, y=df['organ']), 1):
        df.loc[training_idx, f'fold{fold}'] = 0
        df.loc[validation_idx, f'fold{fold}'] = 1
        df[f'fold{fold}'] = df[f'fold{fold}'].astype(np.uint8)

    df['fold6'] = 0

    if verbose:
        logging.info(f'Dataset split into {n_splits} folds')
        for fold in range(1, n_splits + 1):
            df_fold = df[df[f'fold{fold}'] == 1]
            stratify_column_value_counts = df_fold['organ'].value_counts().to_dict()
            annotation_count = df_fold['n_polygons'].sum()
            logging.info(f'Fold {fold} {df_fold.shape} - {stratify_column_value_counts} - {int(annotation_count)} Annotations')

    return df


if __name__ == '__main__':

    df_train = pd.read_csv(settings.DATA / 'train_metadata.csv')
    logging.info(f'Training Set Shape: {df_train.shape} - Memory Usage: {df_train.memory_usage().sum() / 1024 ** 2:.2f} MB')

    get_folds(
        df=df_train,
        n_splits=5,
        shuffle=True,
        random_state=3,
        verbose=True
    )

    df_train[['id'] + [column for column in df_train.columns if column.startswith('fold')]].to_csv(settings.DATA / 'folds.csv', index=False)
    logging.info(f'folds.csv is saved to {settings.DATA}')
