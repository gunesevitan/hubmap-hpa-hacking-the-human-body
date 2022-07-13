import numpy as np
import torch


def soft_predictions_to_labels(x, threshold):

    """
    Convert soft predictions into hard labels in given array

    Parameters
    ----------
    x (array-like of any shape): Soft predictions array
    threshold (float): Threshold for converting soft predictions into hard labels (0 <= threshold <= 1)

    Returns
    -------
    x (array-like of any shape): Hard labels array
    """

    if isinstance(x, torch.Tensor):
        x = x.numpy()
    else:
        x = np.array(x)

    x = np.uint8(x >= threshold)

    return x


def binary_dice_coefficient(ground_truth, predictions, threshold=0.5, eps=0.00001):

    """
    Calculate dice coefficient on given ground truth and predictions arrays

    Parameters
    ----------
    ground_truth (array-like of shape (batch_size, height, width) or (height, width)): Ground truth array
    predictions (array-like of shape (batch_size, height, width) or (height, width)): Predictions array
    threshold (float): Threshold for converting soft predictions into hard labels (0 <= threshold <= 1)
    eps (float): A small number for avoiding division by zero

    Returns
    -------
    dice_coefficient (float): Calculated dice coefficient
    """

    if isinstance(ground_truth, torch.Tensor):
        ground_truth = ground_truth.numpy().astype(np.float32)
    else:
        ground_truth = np.array(ground_truth).astype(np.float32)

    if isinstance(predictions, torch.Tensor):
        predictions = predictions.numpy().astype(np.float32)
    else:
        predictions = np.array(predictions).astype(np.float32)

    if len(ground_truth.shape) != len(predictions.shape):
        raise ValueError('Shape mismatch')

    predictions = soft_predictions_to_labels(x=predictions, threshold=threshold).astype(np.float32)

    if len(ground_truth.shape) == 2:
        # Calculate dice coefficient for single data point
        intersection = np.sum(ground_truth * predictions)
        denominator = np.sum(ground_truth) + np.sum(predictions)
        dice_coefficient = (2 * intersection + eps) / (denominator + eps)
    elif len(ground_truth.shape) == 3:
        # Calculate dice coefficient for batch of data points (assuming first dimension is batch)
        intersection = np.sum(ground_truth * predictions, axis=(1, 2))
        denominator = np.sum(ground_truth, axis=(1, 2)) + np.sum(predictions, axis=(1, 2))
        dice_coefficient = np.mean((2 * intersection + eps) / (denominator + eps))
    else:
        raise ValueError('Invalid shape')

    return float(dice_coefficient)


def binary_intersection_over_union(ground_truth, predictions, threshold=0.5, eps=0.00001):

    """
    Calculate intersection over union on given ground truth and predictions arrays

    Parameters
    ----------
    ground_truth (array-like of shape (batch_size, height, width) or (height, width)): Ground truth array
    predictions (array-like of shape (batch_size, height, width) or (height, width)): Predictions array
    threshold (float): Threshold for converting soft predictions into hard labels (0 <= threshold <= 1)
    eps (float): A small number for avoiding division by zero

    Returns
    -------
    intersection_over_union (float): Calculated intersection over union
    """

    if isinstance(ground_truth, torch.Tensor):
        ground_truth = ground_truth.numpy().astype(np.float32)
    else:
        ground_truth = np.array(ground_truth).astype(np.float32)

    if isinstance(predictions, torch.Tensor):
        predictions = predictions.numpy().astype(np.float32)
    else:
        predictions = np.array(predictions).astype(np.float32)

    if len(ground_truth.shape) != len(predictions.shape):
        raise ValueError('Shape mismatch')

    predictions = soft_predictions_to_labels(x=predictions, threshold=threshold).astype(np.float32)

    if len(ground_truth.shape) == 2:
        # Calculate intersection over union for single data point
        intersection = np.sum(ground_truth * predictions)
        union = np.sum(ground_truth + predictions - ground_truth * predictions)
        intersection_over_union = (intersection + eps) / (union + eps)
    elif len(ground_truth.shape) == 3:
        # Calculate intersection over union for batch of data points (assuming first dimension is batch)
        intersection = np.sum(ground_truth * predictions, axis=(1, 2))
        union = np.sum(ground_truth + predictions - ground_truth * predictions, axis=(1, 2))
        intersection_over_union = np.mean((intersection + eps) / (union + eps))
    else:
        raise ValueError('Invalid shape')

    return float(intersection_over_union)


def mean_binary_dice_coefficient(ground_truth, predictions, thresholds=(0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7), eps=0.00001):

    """
    Calculate dice coefficients on given ground truth and predictions arrays using different thresholds and average them

    Parameters
    ----------
    ground_truth (array-like of shape (batch_size, height, width) or (height, width)): Ground truth array
    predictions (array-like of shape (batch_size, height, width) or (height, width)): Predictions array
    thresholds (tuple of shape (n_thresholds)): Thresholds for converting soft predictions into hard labels (0 <= threshold <= 1)
    eps (float): A small number for avoiding division by zero

    Returns
    -------
    dice_coefficients (dict): Calculated dice coefficients using given thresholds
    mean_dice_coefficient (float): Average of calculated dice coefficients
    """

    dice_coefficients = {}
    for threshold in thresholds:
        dice_coefficients[threshold] = binary_dice_coefficient(ground_truth=ground_truth, predictions=predictions, threshold=threshold, eps=eps)

    mean_dice_coefficient = float(np.mean(list(dice_coefficients.values())))

    return dice_coefficients, mean_dice_coefficient


def mean_binary_intersection_over_union(ground_truth, predictions, thresholds=(0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7), eps=0.00001):

    """
    Calculate intersection over unions on given ground truth and predictions arrays using different thresholds and average them

    Parameters
    ----------
    ground_truth (array-like of shape (batch_size, height, width) or (height, width)): Ground truth array
    predictions (array-like of shape (batch_size, height, width) or (height, width)): Predictions array
    thresholds (tuple of shape (n_thresholds)): Thresholds for converting soft predictions into hard labels (0 <= threshold <= 1)
    eps (float): A small number for avoiding division by zero

    Returns
    -------
    intersection_over_unions (dict): Calculated intersection over unions using given thresholds
    mean_intersection_over_union (float): Average of calculated intersection over unions
    """

    intersection_over_unions = {}
    for threshold in thresholds:
        intersection_over_unions[threshold] = binary_intersection_over_union(ground_truth=ground_truth, predictions=predictions, threshold=threshold, eps=eps)

    mean_intersection_over_union = float(np.mean(list(intersection_over_unions.values())))

    return intersection_over_unions, mean_intersection_over_union
