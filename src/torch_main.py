import argparse
import yaml
import pandas as pd

import settings
import tabular_preprocessing
from torch_trainers import SemanticSegmentationTrainer


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('config_path', type=str)
    parser.add_argument('mode', type=str)
    args = parser.parse_args()

    config = yaml.load(open(args.config_path, 'r'), Loader=yaml.FullLoader)
    df_train = pd.read_csv(settings.DATA / 'train_metadata.csv')
    df_test = pd.read_csv(settings.DATA / 'test_metadata.csv')
    df_folds = pd.read_csv(settings.DATA / 'folds.csv')
    external_data = {
        dataset: pd.read_csv(settings.DATA / 'external_data' / dataset / 'metadata.csv')
        for dataset in config['dataset_parameters']['external_data']
    }

    df_train, df_test = tabular_preprocessing.preprocess_datasets(
        df_train=df_train,
        df_test=df_test,
        df_folds=df_folds,
        external_data=external_data
    )

    if config['task'] == 'semantic_segmentation':

        trainer = SemanticSegmentationTrainer(
            dataset_parameters=config['dataset_parameters'],
            model_parameters=config['model_parameters'],
            training_parameters=config['training_parameters'],
            transform_parameters=config['transform_parameters'],
            inference_parameters=config['inference_parameters'],
            persistence_parameters=config['persistence_parameters']
        )

    if args.mode == 'train':
        trainer.train_and_validate(df_train=df_train, df_test=df_test)
    elif args.mode == 'inference':
        trainer.inference(df_train=df_train)
