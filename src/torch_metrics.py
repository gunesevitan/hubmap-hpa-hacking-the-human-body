import torch


def dice_coefficient(ground_truth, predictions, rounding_threshold=0.5, eps=0.001):

    """
    Calculate dice coefficient on given ground truth and predictions

    Parameters
    ----------
    ground_truth (torch.FloatTensor of shape (batch_size, 1, height, width)): Ground truth tensor
    predictions (torch.FloatTensor of shape (batch_size, 1, height, width)): Predictions tensor
    rounding_threshold (float): Threshold for rounding soft predictions into labels
    eps (float): A small number in order to avoid division by zero

    Returns
    -------
    dice (Torch.FloatTensor): Dice coefficient
    """

    ground_truth = ground_truth.to(torch.float32)
    predictions = (predictions > rounding_threshold).to(torch.float32)
    intersection = (ground_truth * predictions).sum(dim=(2, 3))
    denominator = ground_truth.sum(dim=(2, 3)) + predictions.sum(dim=(2, 3))
    dice = float(((2 * intersection + eps) / (denominator + eps)).mean(dim=(1, 0)))

    return dice


def intersection_over_union(ground_truth, predictions, rounding_threshold=0.5, eps=0.001):

    """
    Calculate intersection over union on given ground truth and predictions

    Parameters
    ----------
    ground_truth (torch.FloatTensor of shape (batch_size, 3, height, width)): Ground truth tensor
    predictions (torch.FloatTensor of shape (batch_size, 3, height, width)): Predictions tensor
    rounding_threshold (float): Threshold for rounding soft predictions into labels
    eps (float): A small number in order to avoid division by zero

    Returns
    -------
    iou (Torch.FloatTensor): Intersection over union
    """

    ground_truth = ground_truth.to(torch.float32)
    predictions = (predictions > rounding_threshold).to(torch.float32)
    intersection = (ground_truth * predictions).sum(dim=(2, 3))
    union = (ground_truth + predictions - ground_truth * predictions).sum(dim=(2, 3))
    iou = float(((intersection + eps) / (union + eps)).mean(dim=(1, 0)))

    return iou
