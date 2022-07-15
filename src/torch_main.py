import argparse
import yaml
import pandas as pd

import settings
from torch_trainers import SemanticSegmentationTrainer


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('config_path', type=str)
    parser.add_argument('mode', type=str)
    args = parser.parse_args()

    config = yaml.load(open(args.config_path, 'r'), Loader=yaml.FullLoader)
    df_train = pd.read_csv(settings.DATA / 'train_metadata.csv')
    df_train = df_train.merge(pd.read_csv(settings.DATA / 'folds.csv'), on='id', how='left')
    df_test = pd.read_csv(settings.DATA / 'test_metadata.csv')

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
