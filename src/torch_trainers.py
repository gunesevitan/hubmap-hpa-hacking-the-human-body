import logging
from pathlib import Path
import json
import pandas as pd
import tifffile
from tqdm import tqdm
import numpy as np
import cv2
import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
import torch.optim as optim
import torch.optim.swa_utils as swa_utils

import settings
import annotation_utils
import visualization
import transforms
import torch_datasets
import torch_modules
import torch_utils
import metrics
import evaluation
import test_time_augmentations


class SemanticSegmentationTrainer:

    def __init__(self, dataset_parameters, model_parameters, training_parameters, transform_parameters, inference_parameters, persistence_parameters):

        self.dataset_parameters = dataset_parameters
        self.model_parameters = model_parameters
        self.training_parameters = training_parameters
        self.transform_parameters = transform_parameters
        self.inference_parameters = inference_parameters
        self.persistence_parameters = persistence_parameters

    def train(self, train_loader, model, criterion, optimizer, device, scheduler=None):

        """
        Train given model on given data loader

        Parameters
        ----------
        train_loader (torch.utils.data.DataLoader): Training set data loader
        model (torch.nn.Module): Model to train
        criterion (torch.nn.Module): Loss function
        optimizer (torch.optim.Optimizer): Optimizer
        device (torch.device): Location of the model and inputs
        scheduler (torch.optim.LRScheduler or None): Learning rate scheduler

        Returns
        -------
        train_loss (float): Average training loss after model is fully trained on training set data loader
        """

        model.train()
        progress_bar = tqdm(train_loader)
        losses = []

        for inputs, targets in progress_bar:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            if scheduler is not None:
                scheduler.step()
            losses.append(loss.detach().item())
            average_loss = np.mean(losses)
            lr = scheduler.get_last_lr()[0] if scheduler is not None else optimizer.param_groups[0]['lr']
            progress_bar.set_description(f'train_loss: {average_loss:.6f} - lr: {lr:.8f}')

        train_loss = np.mean(losses)
        return train_loss

    def validate(self, val_loader, model, criterion, device):

        """
        Validate given model on given data loader

        Parameters
        ----------
        val_loader (torch.utils.data.DataLoader): Validation set data loader
        model (torch.nn.Module): Model to validate
        criterion (torch.nn.Module): Loss function
        device (torch.device): Location of the model and inputs

        Returns
        -------
        val_loss (float): Average validation loss after model is fully validated on validation set data loader
        val_dice_coefficients (tuple and float): Validation dice coefficients after model is fully validated on validation set data loader
        val_intersection_over_unions (tuple and float): Validation intersection over unions after model is fully validated on validation set data loader
        """

        model.eval()
        progress_bar = tqdm(val_loader)
        losses = []
        ground_truth = []
        predictions = []

        with torch.no_grad():
            for inputs, targets in progress_bar:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                losses.append(loss.detach().item())
                average_loss = np.mean(losses)
                progress_bar.set_description(f'val_loss: {average_loss:.6f}')
                ground_truth += [(targets.detach().cpu())]
                predictions += [(outputs.detach().cpu())]

        val_loss = np.mean(losses)
        ground_truth = torch.squeeze(torch.cat(ground_truth, dim=0), dim=1)
        predictions = torch.sigmoid(torch.squeeze(torch.cat(predictions, dim=0), dim=1))
        val_dice_coefficients = metrics.mean_binary_dice_coefficient(ground_truth=ground_truth, predictions=predictions, thresholds=self.inference_parameters['label_threshold_range'])
        val_intersection_over_unions = metrics.mean_binary_intersection_over_union(ground_truth=ground_truth, predictions=predictions, thresholds=self.inference_parameters['label_threshold_range'])

        return val_loss, val_dice_coefficients, val_intersection_over_unions

    def train_and_validate(self, df_train, df_test):

        """
        Train and validate on inputs and targets listed on given dataframes with specified configuration and transforms

        Parameters
        ----------
        df_train (pandas.DataFrame of shape (n_rows, n_columns)): Dataframe of filenames, targets and folds
        df_test (pandas.DataFrame of shape (n_rows, n_columns)): Dataframe of filenames
        """

        logging.info(f'\n{"-" * 30}\nRunning {self.persistence_parameters["name"]} Model for Training - Seed: {self.training_parameters["random_state"]}\n{"-" * 30}\n')

        # Create directory for models and visualizations
        model_root_directory = Path(settings.MODELS / self.persistence_parameters['name'])
        model_root_directory.mkdir(parents=True, exist_ok=True)

        dataset_transforms = transforms.get_semantic_segmentation_transforms(**self.transform_parameters)
        scores = []

        for fold in self.training_parameters['folds']:

            train_idx, val_idx = df_train.loc[df_train[fold] == 0].index, df_train.loc[df_train[fold] == 1].index
            # Validate on training set if validation is set is not specified
            if len(val_idx) == 0:
                val_idx = train_idx

            logging.info(f'\n{fold} - Training: {len(train_idx)} ({len(train_idx) // self.training_parameters["training_batch_size"] + 1} steps) - Validation {len(val_idx)} ({len(val_idx) // self.training_parameters["test_batch_size"] + 1} steps)')
            train_dataset = torch_datasets.SemanticSegmentationDataset(
                image_paths=df_train.loc[train_idx, self.dataset_parameters['inputs']].values,
                organs=df_train.loc[train_idx, 'organ'].values,
                data_sources=df_train.loc[train_idx, 'data_source'].values,
                masks=df_train.loc[train_idx, 'rle'].values,
                transforms=dataset_transforms['train'],
                imaging_measurement_adaptation_probability=self.transform_parameters['imaging_measurement_adaptation_probability'],
                standardize_luminosity_probability=self.transform_parameters['standardize_luminosity_probability']
            )
            train_loader = DataLoader(
                train_dataset,
                batch_size=self.training_parameters['training_batch_size'],
                sampler=RandomSampler(train_dataset),
                pin_memory=False,
                drop_last=False,
                num_workers=self.training_parameters['num_workers']
            )
            val_dataset = torch_datasets.SemanticSegmentationDataset(
                image_paths=df_train.loc[val_idx, self.dataset_parameters['inputs']].values,
                organs=df_train.loc[val_idx, 'organ'].values,
                data_sources=df_train.loc[val_idx, 'data_source'].values,
                masks=df_train.loc[val_idx, 'rle'].values,
                transforms=dataset_transforms['val'],
                imaging_measurement_adaptation_probability=0,
                standardize_luminosity_probability=0
            )
            val_loader = DataLoader(
                val_dataset,
                batch_size=self.training_parameters['test_batch_size'],
                sampler=SequentialSampler(val_dataset),
                pin_memory=False,
                drop_last=False,
                num_workers=self.training_parameters['num_workers']
            )

            # Set model, loss function, device and seed for reproducible results
            torch_utils.set_seed(self.training_parameters['random_state'], deterministic_cudnn=self.training_parameters['deterministic_cudnn'])
            device = torch.device(self.training_parameters['device'])
            criterion = getattr(torch_modules, self.training_parameters['loss_function'])(**self.training_parameters['loss_args'])

            if self.model_parameters['model_module'] in ['smp', 'monai']:
                model = torch_modules.SemanticSegmentationModel(
                    self.model_parameters['model_module'],
                    self.model_parameters['model_class'],
                    self.model_parameters['model_args']
                )
            elif self.model_parameters['model_module'] == 'transformers':
                model = torch_modules.HuggingFaceTransformersModel(
                    self.model_parameters['model_class'],
                    self.model_parameters['model_args'],
                    self.model_parameters['upsample_args']
                )

            if self.model_parameters['model_checkpoint_path'] is not None:
                model.load_state_dict(torch.load(self.model_parameters['model_checkpoint_path']))
            model.to(device)

            # Set optimizer, learning rate scheduler and stochastic weight averaging
            optimizer = getattr(optim, self.training_parameters['optimizer'])(model.parameters(), **self.training_parameters['optimizer_args'])
            scheduler = getattr(optim.lr_scheduler, self.training_parameters['lr_scheduler'])(optimizer, **self.training_parameters['lr_scheduler_args'])
            if self.training_parameters['swa_start_epoch'] > 0:
                swa_model = swa_utils.AveragedModel(model, device=device)
                swa_scheduler = swa_utils.SWALR(
                    optimizer,
                    swa_lr=self.training_parameters['swa_lr'],
                    anneal_epochs=self.training_parameters['swa_anneal_epochs'],
                    anneal_strategy=self.training_parameters['swa_anneal_strategy'],
                    last_epoch=-1
                )
            else:
                swa_model = None
                swa_scheduler = None

            early_stopping = False
            summary = {
                'train_loss': [],
                'val_loss': [],
                'val_dice_coefficient': [],
                'val_intersection_over_union': []
            }

            for epoch in range(1, self.training_parameters['epochs'] + 1):

                if early_stopping:
                    break

                if self.training_parameters['lr_scheduler'] == 'ReduceLROnPlateau':
                    # Step on validation loss if learning rate scheduler is ReduceLROnPlateau
                    train_loss = self.train(train_loader, model, criterion, optimizer, device, scheduler=None)
                    val_loss, val_dice_coefficients, val_intersection_over_unions = self.validate(val_loader, model, criterion, device)
                    scheduler.step(val_loss)
                else:
                    # Learning rate scheduler works in training function if it is not ReduceLROnPlateau
                    train_loss = self.train(train_loader, model, criterion, optimizer, device, scheduler)
                    val_loss, val_dice_coefficients, val_intersection_over_unions = self.validate(val_loader, model, criterion, device)

                if self.training_parameters['swa_start_epoch'] > 0:
                    if epoch >= self.training_parameters['swa_start_epoch']:
                        swa_model.update_parameters(model)
                        swa_scheduler.step()

                logging.info(
                    f'''
                    Epoch {epoch} - Training Loss: {train_loss:.4f} Validation Loss: {val_loss:.4f}
                    Dice Coefficients: {val_dice_coefficients[0]} (Mean Dice Coefficient {val_dice_coefficients[1]:.4f})
                    Intersection over Unions: {val_intersection_over_unions[0]} (Mean Intersection over Union {val_intersection_over_unions[1]:.4f})
                    '''
                )

                if epoch in self.persistence_parameters['save_epoch_model']:
                    # Save model if current epoch is specified to be saved
                    torch.save(model.state_dict(), model_root_directory / f'model_{fold}_epoch_{epoch}.pt')
                    logging.info(f'Saved model_{fold}_epoch_{epoch}.pt to {model_root_directory}')

                best_val_loss = np.min(summary['val_loss']) if len(summary['val_loss']) > 0 else np.inf
                if val_loss < best_val_loss:
                    # Save model if validation loss improves
                    if self.persistence_parameters['save_best_model']:
                        torch.save(model.state_dict(), model_root_directory / f'model_{fold}_best.pt')
                        logging.info(f'Saved model_{fold}_best.pt to {model_root_directory} (validation loss decreased from {best_val_loss:.6f} to {val_loss:.6f})')

                    # Save epoch predictions visualizations if validation loss improves
                    if self.persistence_parameters['visualize_epoch_predictions']:

                        model.eval()

                        # Create directory for epoch predictions visualizations
                        epoch_predictions_directory = Path(model_root_directory / 'epoch_predictions')
                        epoch_predictions_directory.mkdir(parents=True, exist_ok=True)

                        # Sample single image for every organ type from training set with fixed random seed for evaluating epochs
                        np.random.seed(self.training_parameters['random_state'])
                        df_evaluation = pd.concat((
                            df_train.loc[val_idx, :].groupby('organ').sample(1),
                            df_test
                        ), ignore_index=True, axis=0)

                        for idx, row in df_evaluation.iterrows():

                            if row['data_source'] == 'HPA' or row['data_source'] == 'Hubmap':
                                evaluation_image = tifffile.imread(row['image_filename'])
                            elif row['data_source'] == 'GTEx':
                                evaluation_image = cv2.imread(row['image_filename'])
                            else:
                                raise ValueError(f'Invalid data source: {row["data_source"]}')

                            if idx != (df_evaluation.shape[0] - 1):
                                evaluation_ground_truth_mask = annotation_utils.decode_rle_mask(rle_mask=row['rle'], shape=evaluation_image.shape[:2])
                                if row['data_source'] == 'Hubmap' or row['data_source'] == 'HPA':
                                    evaluation_ground_truth_mask = evaluation_ground_truth_mask.T
                            else:
                                evaluation_ground_truth_mask = None

                            evaluation_inputs = dataset_transforms['val'](image=evaluation_image)['image'].float()
                            evaluation_inputs = evaluation_inputs.to(device)

                            with torch.no_grad():
                                evaluation_outputs = model(torch.unsqueeze(evaluation_inputs, dim=0))

                            evaluation_predictions_mask = torch.sigmoid(torch.squeeze(torch.squeeze(evaluation_outputs.detach().cpu(), dim=0), dim=0)).numpy().astype(np.float32)
                            # Resize evaluation predictions mask back to its original size and evaluate it on multiple thresholds
                            evaluation_predictions_mask = cv2.resize(evaluation_predictions_mask, (evaluation_image.shape[1], evaluation_image.shape[0]), interpolation=cv2.INTER_CUBIC)
                            evaluation_summary = evaluation.evaluate_predictions(
                                ground_truth=evaluation_ground_truth_mask,
                                predictions=evaluation_predictions_mask,
                                threshold=self.inference_parameters['label_thresholds'][row['data_source']][row['organ']],
                                thresholds=self.inference_parameters['label_threshold_range']
                            )

                            # Convert evaluation predictions mask's soft predictions to labels and visualize it
                            evaluation_predictions_mask = metrics.soft_predictions_to_labels(x=evaluation_predictions_mask, threshold=self.inference_parameters['label_thresholds'][row['data_source']][row['organ']])
                            visualization.visualize_predictions(
                                image=evaluation_image,
                                ground_truth=evaluation_ground_truth_mask,
                                predictions=evaluation_predictions_mask,
                                metadata=row.to_dict(),
                                evaluation_summary=evaluation_summary,
                                path=epoch_predictions_directory / f'{row["id"]}_{row["organ"]}_{fold}_epoch{epoch}_{val_loss:.4f}_predictions.png'
                            )

                        logging.info(f'Saved {fold} epoch {epoch} predictions to {epoch_predictions_directory}')

                summary['train_loss'].append(train_loss)
                summary['val_loss'].append(val_loss)
                summary['val_dice_coefficient'].append(np.median(list(val_dice_coefficients[0].values())))
                summary['val_intersection_over_union'].append(np.median(list(val_intersection_over_unions[0].values())))

                best_epoch = np.argmin(summary['val_loss'])
                if self.training_parameters['early_stopping_patience'] > 0:
                    # Trigger early stopping if early stopping patience is greater than 0
                    if len(summary['val_loss']) - best_epoch >= self.training_parameters['early_stopping_patience']:
                        logging.info(
                            f'''
                            Early Stopping (validation loss didn\'t improve for {self.training_parameters["early_stopping_patience"]} epochs)
                            Best Epoch ({best_epoch + 1}) Validation Loss: {summary["val_loss"][best_epoch]:.6f} Dice Coefficient: {summary["val_dice_coefficient"][best_epoch]:.4f}  Intersection over Union: {summary["val_intersection_over_union"][best_epoch]:.4f}
                            '''
                        )
                        early_stopping = True
                        scores.append({
                            'val_loss': summary['val_loss'][best_epoch],
                            'val_dice_coefficient': summary['val_dice_coefficient'][best_epoch],
                            'val_intersection_over_union': summary['val_intersection_over_union'][best_epoch]
                        })
                else:
                    if epoch == self.training_parameters['epochs']:
                        scores.append({
                            'val_loss': summary['val_loss'][-1],
                            'val_dice_coefficient': summary['val_dice_coefficient'][-1],
                            'val_intersection_over_union': summary['val_intersection_over_union'][-1]
                        })

            if self.persistence_parameters['visualize_learning_curve']:
                visualization.visualize_learning_curve(
                    training_losses=summary['train_loss'],
                    validation_losses=summary['val_loss'],
                    path=model_root_directory / f'learning_curve_{fold}.png'
                )
                logging.info(f'Saved learning_curve_{fold}.png to {model_root_directory}')

            if self.training_parameters['swa_start_epoch'] > 0:
                # Perform one pass over data to estimate the activation statistics for batch normalization layers in the model
                swa_utils.update_bn(train_loader, swa_model, device=device)

        df_scores = pd.DataFrame(scores)
        for score_idx, row in df_scores.iterrows():
            logging.info(f'Fold {int(score_idx) + 1} - Validation Scores: {json.dumps(row.to_dict(), indent=2)}')
        logging.info(f'\nMean Validation Scores: {json.dumps(df_scores.mean(axis=0).to_dict(), indent=2)} (±{json.dumps(df_scores.std(axis=0).to_dict(), indent=2)})')

        if self.persistence_parameters['visualize_training_scores']:
            visualization.visualize_scores(
                df_scores=df_scores,
                path=model_root_directory / f'training_scores.png'
            )
            logging.info(f'Saved training_scores.png to {model_root_directory}')

    def inference(self, df_train):

        """
        Inference on inputs and targets listed on given dataframes with specified configuration and transforms

        Parameters
        ----------
        df_train (pandas.DataFrame of shape (n_rows, n_columns)): Dataframe of filenames, targets and folds
        """

        logging.info(f'\n{"-" * 30}\nRunning {self.persistence_parameters["name"]} Model for Inference - Seed: {self.training_parameters["random_state"]}\n{"-" * 30}\n')

        # Create directory for models and visualizations
        model_root_directory = Path(settings.MODELS / self.persistence_parameters['name'])
        model_root_directory.mkdir(parents=True, exist_ok=True)
        # Create directory for final predictions visualizations
        final_predictions_directory = Path(model_root_directory / 'final_predictions')
        final_predictions_directory.mkdir(parents=True, exist_ok=True)

        dataset_transforms = transforms.get_semantic_segmentation_transforms(**self.transform_parameters)

        for fold in self.inference_parameters['folds']:

            val_idx = df_train.loc[df_train[fold] == 1].index
            logging.info(f'\n{fold}  - Validation {len(val_idx)} ({len(val_idx) // self.training_parameters["test_batch_size"] + 1} steps)')

            # Set model, loss function, device and seed for reproducible results
            torch_utils.set_seed(self.training_parameters['random_state'], deterministic_cudnn=self.training_parameters['deterministic_cudnn'])
            device = torch.device(self.training_parameters['device'])
            if self.model_parameters['model_module'] in ['smp', 'monai']:
                model = torch_modules.SemanticSegmentationModel(
                    self.model_parameters['model_module'],
                    self.model_parameters['model_class'],
                    self.model_parameters['model_args']
                )
            elif self.model_parameters['model_module'] == 'transformers':
                model = torch_modules.HuggingFaceTransformersModel(
                    self.model_parameters['model_class'],
                    self.model_parameters['model_args'],
                    self.model_parameters['upsample_args']
                )
            elif self.model_parameters['model_module'] == 'coat_daformer':
                model = torch_modules.CoaTDAFormer(self.model_parameters['model_args'])
            else:
                raise ValueError('Invalid Model Module')

            model.load_state_dict(torch.load(model_root_directory / f'model_{fold}_best.pt'))
            model.to(device)
            model.eval()

            for idx, row in tqdm(df_train.loc[val_idx, :].iterrows(), total=len(val_idx)):

                if row['data_source'] == 'HPA' or row['data_source'] == 'Hubmap':
                    image = tifffile.imread(row['image_filename'])
                elif row['data_source'] == 'GTEx':
                    image = cv2.imread(row['image_filename'])
                else:
                    raise ValueError(f'Invalid data source: {row["data_source"]}')

                image_resized = cv2.resize(
                    image,
                    dsize=self.inference_parameters['size'][row['data_source']][row['organ']],
                    interpolation=cv2.INTER_CUBIC
                )

                if self.inference_parameters['tta']:
                    # Stack augmented images on batch dimension
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
                else:
                    inputs = torch.unsqueeze(dataset_transforms['test'](image=image_resized)['image'], dim=0)

                inputs = inputs.to('cuda')

                with torch.no_grad():
                    outputs = model(inputs)

                predictions_mask = outputs.detach().cpu()
                predictions_mask = torch.sigmoid(torch.squeeze(predictions_mask, dim=1)).numpy().astype(np.float32)

                if self.inference_parameters['tta']:
                    # Apply inverse of test-time augmentations and aggregate predictions
                    predictions_mask[1, :, :] = test_time_augmentations.horizontal_flip(predictions_mask[1, :, :])
                    predictions_mask[2, :, :] = test_time_augmentations.vertical_flip(predictions_mask[2, :, :])
                    predictions_mask[3, :, :] = test_time_augmentations.horizontal_flip(test_time_augmentations.vertical_flip(predictions_mask[3, :, :]))
                    predictions_mask = np.mean(predictions_mask, axis=0)

                # Decode RLE mask string into 2d binary semantic segmentation mask array
                ground_truth_mask = annotation_utils.decode_rle_mask(rle_mask=row['rle'], shape=image.shape[:2])
                if row['data_source'] == 'Hubmap' or row['data_source'] == 'HPA':
                    ground_truth_mask = ground_truth_mask.T

                # Resize predictions mask back to its original size and evaluate it
                predictions_mask = cv2.resize(predictions_mask, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_CUBIC)
                predictions_evaluation_summary = evaluation.evaluate_predictions(
                    ground_truth=ground_truth_mask,
                    predictions=predictions_mask,
                    threshold=self.inference_parameters['label_thresholds'][row['data_source']][row['organ']],
                    thresholds=self.inference_parameters['label_threshold_range'])

                if self.persistence_parameters['evaluate_final_predictions']:
                    with open(final_predictions_directory / f'{row["id"]}_evaluation.json', mode='w') as f:
                        json.dump(predictions_evaluation_summary, f, indent=2)

                df_train.loc[df_train['id'] == row['id'], 'dice_coefficient'] = predictions_evaluation_summary['scores']['dice_coefficient']
                df_train.loc[df_train['id'] == row['id'], 'intersection_over_union'] = predictions_evaluation_summary['scores']['intersection_over_union']

                try:
                    label_threshold = self.inference_parameters['label_thresholds'][row['data_source']][row['organ']]
                except KeyError:
                    # Set label threshold to 0.1 for unseen organs or data sources
                    label_threshold = 0.1

                # Convert evaluation predictions mask's soft predictions to labels
                predictions_mask = metrics.soft_predictions_to_labels(x=predictions_mask, threshold=label_threshold)

                if self.persistence_parameters['visualize_final_predictions']:
                    visualization.visualize_predictions(
                        image=image,
                        ground_truth=ground_truth_mask,
                        predictions=predictions_mask,
                        metadata=row.to_dict(),
                        evaluation_summary=predictions_evaluation_summary,
                        path=final_predictions_directory / f'{row["id"]}_{row["organ"]}_predictions.png'
                    )

            logging.info(f'Saved predictions evaluation summaries and predictions visualizations to {final_predictions_directory}')

        scores_evaluation_summary = evaluation.evaluate_scores(df=df_train, folds=self.inference_parameters['folds'])
        logging.info(json.dumps(scores_evaluation_summary, indent=2))
        with open(model_root_directory / f'inference_scores.json', mode='w') as f:
            json.dump(scores_evaluation_summary, f, indent=2)

        logging.info(f'Saved inference_scores.json to {model_root_directory}')
