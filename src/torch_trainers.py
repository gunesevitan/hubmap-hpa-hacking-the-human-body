import logging
from pathlib import Path
import json
import pandas as pd
from tqdm import tqdm
import numpy as np
import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
import torch.optim as optim

import settings
import visualization
import transforms
import torch_datasets
import torch_modules
import torch_utils
import torch_metrics


class SemanticSegmentationTrainer:

    def __init__(self, dataset_parameters, model_parameters, training_parameters, transform_parameters, post_processing_parameters, persistence_parameters):

        self.dataset_parameters = dataset_parameters
        self.model_parameters = model_parameters
        self.training_parameters = training_parameters
        self.transform_parameters = transform_parameters
        self.post_processing_parameters = post_processing_parameters
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
        val_dice_coefficient (float): Validation dice coefficient after model is fully validated on validation set data loader
        val_iou (float): Validation intersection over union after model is fully validated on validation set data loader
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
        ground_truth = torch.cat(ground_truth, dim=0)
        predictions = torch.sigmoid(torch.cat(predictions, dim=0))
        val_dice_coefficient = torch_metrics.dice_coefficient(ground_truth=ground_truth, predictions=predictions, rounding_threshold=0.5)
        val_iou = torch_metrics.intersection_over_union(ground_truth=ground_truth, predictions=predictions, rounding_threshold=0.5)

        return val_loss, val_dice_coefficient, val_iou

    def train_and_validate(self, df_train):

        """
        Train and validate on inputs and targets listed on given dataframe with specified configuration and transforms

        Parameters
        ----------
        df_train (pandas.DataFrame of shape (n_rows, n_columns)): Dataframe of filenames, targets and folds
        """

        logging.info(f'\n{"-" * 30}\nRunning {self.persistence_parameters["name"]} Model for Training - Seed: {self.training_parameters["random_state"]}\n{"-" * 30}\n')

        # Create directory for models, visualizations and predictions
        model_directory = Path(settings.MODELS / self.persistence_parameters['name'])
        model_directory.mkdir(parents=True, exist_ok=True)

        dataset_transforms = transforms.get_semantic_segmentation_transforms(**self.transform_parameters)

        # Calculate and collect scores iteratively
        scores = []

        for fold in self.training_parameters['folds']:

            train_idx, val_idx = df_train.loc[df_train[fold] == 0].index, df_train.loc[df_train[fold] == 1].index
            logging.info(f'\n{fold} - Training: {len(train_idx)} ({len(train_idx) // self.training_parameters["training_batch_size"] + 1} steps) - Validation {len(val_idx)} ({len(val_idx) // self.training_parameters["test_batch_size"] + 1} steps)')
            train_dataset = torch_datasets.SemanticSegmentationDataset(
                image_paths=df_train.loc[train_idx, 'image_filename'].values,
                masks=df_train.loc[train_idx, self.dataset_parameters['target_directory']].values,
                transforms=dataset_transforms['train'],
                mask_format=self.dataset_parameters['mask_format']
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
                image_paths=df_train.loc[val_idx, 'image_filename'].values,
                masks=df_train.loc[val_idx, self.dataset_parameters['target_directory']].values,
                transforms=dataset_transforms['val'],
                mask_format=self.dataset_parameters['mask_format']
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
            model = torch_modules.SemanticSegmentationModel(self.model_parameters['model_class'], self.model_parameters['model_args'])
            if self.model_parameters['model_checkpoint_path'] is not None:
                model.load_state_dict(torch.load(self.model_parameters['model_checkpoint_path']))
            model.to(device)

            # Set optimizer and learning rate scheduler
            optimizer = getattr(optim, self.training_parameters['optimizer'])(model.parameters(), **self.training_parameters['optimizer_args'])
            scheduler = getattr(optim.lr_scheduler, self.training_parameters['lr_scheduler'])(optimizer, **self.training_parameters['lr_scheduler_args'])

            early_stopping = False
            summary = {
                'train_loss': [],
                'val_loss': [],
                'val_dice_coefficient': [],
                'val_iou': []
            }

            for epoch in range(1, self.training_parameters['epochs'] + 1):

                if early_stopping:
                    break

                if self.training_parameters['lr_scheduler'] == 'ReduceLROnPlateau':
                    # Step on validation loss if learning rate scheduler is ReduceLROnPlateau
                    train_loss = self.train(train_loader, model, criterion, optimizer, device, scheduler=None)
                    val_loss, val_dice_coefficient, val_iou = self.validate(val_loader, model, criterion, device)
                    scheduler.step(val_loss)
                else:
                    # Learning rate scheduler will work in validation function if it is not ReduceLROnPlateau
                    train_loss = self.train(train_loader, model, criterion, optimizer, device, scheduler)
                    val_loss, val_dice_coefficient, val_iou = self.validate(val_loader, model, criterion, device)

                logging.info(f'Epoch {epoch} - Training Loss: {train_loss:.4f} - Validation Loss: {val_loss:.4f} Dice Coefficient: {val_dice_coefficient:.4f} IoU: {val_iou:.4f}')
                best_val_loss = np.min(summary['val_loss']) if len(summary['val_loss']) > 0 else np.inf
                if val_loss < best_val_loss:
                    # Save model if validation loss improves
                    if self.persistence_parameters['save_models']:
                        torch.save(model.state_dict(), model_directory / f'model_{fold}.pt')
                        logging.info(f'Saved model_{fold}.pt to {model_directory} (validation loss decreased from {best_val_loss:.6f} to {val_loss:.6f})')

                summary['train_loss'].append(train_loss)
                summary['val_loss'].append(val_loss)
                summary['val_dice_coefficient'].append(val_dice_coefficient)
                summary['val_iou'].append(val_iou)

                best_iteration = np.argmin(summary['val_loss'])
                if len(summary['val_loss']) - best_iteration >= self.training_parameters['early_stopping_patience']:
                    logging.info(f'Early stopping (validation loss didn\'t increase for {self.training_parameters["early_stopping_patience"]} epochs/steps)')
                    logging.info(f'Best validation loss is {summary["val_loss"][best_iteration]:.6f}')
                    early_stopping = True

                    scores.append({
                        'val_loss': summary['val_loss'][best_iteration],
                        'val_dice_coefficient': summary['val_dice_coefficient'][best_iteration],
                        'val_iou': summary['val_iou'][best_iteration]
                    })

            if self.persistence_parameters['save_visualizations']:
                visualization.visualize_learning_curve(
                    training_losses=summary['train_loss'],
                    validation_losses=summary['val_loss'],
                    path=model_directory / f'learning_curve_{fold}.png'
                )
                logging.info(f'Saved learning_curve_{fold}.png to {model_directory}')

        df_scores = pd.DataFrame(scores)
        for idx, row in df_scores.iterrows():
            logging.info(f'Fold {int(idx) + 1} - Validation Scores: {json.dumps(row.to_dict(), indent=2)}')
        logging.info(f'\n{self.persistence_parameters["name"]} Mean Validation Scores: {json.dumps(df_scores.mean(axis=0).to_dict(), indent=2)} (±{json.dumps(df_scores.std(axis=0).to_dict(), indent=2)})')

        if self.persistence_parameters['save_visualizations']:
            visualization.visualize_scores(
                df_scores=df_scores,
                path=model_directory / f'scores.png'
            )
            logging.info(f'Saved scores.png to {model_directory}')

    def inference(self, df_train):

        """
        Inference on inputs and targets listed on given dataframe with specified configuration and transforms

        Parameters
        ----------
        df_train (pandas.DataFrame of shape (n_rows, n_columns)): Dataframe of filenames, targets and folds
        """

        logging.info(f'\n{"-" * 30}\nRunning {self.persistence_parameters["name"]} Model for Inference - Seed: {self.training_parameters["random_state"]}\n{"-" * 30}\n')

        # Create directory for models, visualizations and predictions
        model_directory = Path(settings.MODELS / self.persistence_parameters['name'])
        model_directory.mkdir(parents=True, exist_ok=True)

        dataset_transforms = transforms.get_semantic_segmentation_transforms(**self.transform_parameters)

        # Calculate and collect scores iteratively
        scores = []

        for fold_idx, fold in enumerate(self.training_parameters['folds']):

            val_idx = df_train.loc[df_train[fold] == 1].index
            logging.info(f'\n{fold}  - Validation {len(val_idx)} ({len(val_idx) // self.training_parameters["test_batch_size"] + 1} steps)')
            val_dataset = torch_datasets.SemanticSegmentationDataset(
                image_paths=df_train.loc[val_idx, 'image_filename'].values,
                masks=df_train.loc[val_idx, self.dataset_parameters['target_directory']].values,
                transforms=dataset_transforms['val'],
                mask_format=self.dataset_parameters['mask_format']
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
            model = torch_modules.SemanticSegmentationModel(self.model_parameters['model_class'], self.model_parameters['model_args'])
            model.load_state_dict(torch.load(model_directory / f'model_{fold}.pt'))
            model.to(device)

            progress_bar = tqdm(val_loader)
            ground_truth = []
            predictions = []

            with torch.no_grad():
                for inputs, targets in progress_bar:
                    inputs, targets = inputs.to(device), targets.to(device)
                    outputs = model(inputs)
                    ground_truth += [(targets.detach().cpu())]
                    predictions += [(outputs.detach().cpu())]

            ground_truth = torch.cat(ground_truth, dim=0).long()
            predictions = torch.sigmoid(torch.cat(predictions, dim=0))

            val_dice_coefficient = torch_metrics.dice_coefficient(ground_truth, predictions, rounding_threshold=self.post_processing_parameters['rounding_threshold'])
            val_iou = torch_metrics.intersection_over_union(ground_truth, predictions, rounding_threshold=self.post_processing_parameters['rounding_threshold'])

            scores.append({
                'val_dice_coefficient': val_dice_coefficient,
                'val_iou': val_iou
            })
            logging.info(f'{fold} - Validation Scores: {json.dumps(scores[fold_idx], indent=2)}')

        df_scores = pd.DataFrame(scores)
        for idx, row in df_scores.iterrows():
            logging.info(f'Fold {int(idx) + 1} - Validation Scores: {json.dumps(row.to_dict(), indent=2)}')
        logging.info(f'\n{self.persistence_parameters["name"]} Mean Validation Scores: {json.dumps(df_scores.mean(axis=0).to_dict(), indent=2)} (±{json.dumps(df_scores.std(axis=0).to_dict(), indent=2)})')

        if self.persistence_parameters['save_visualizations']:
            visualization.visualize_scores(
                df_scores=df_scores,
                path=model_directory / f'scores.png'
            )
            logging.info(f'Saved scores.png to {model_directory}')
