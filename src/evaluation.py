import numpy as np
import cv2

import metrics


def extract_spatial_properties(mask):

    """
    Extract spatial properties from given mask

    Parameters
    ----------
    mask (numpy.ndarray of shape (height, width)): 2d segmentation mask

    Returns
    -------
    spatial_properties (dict): Dictionary of spatial properties extracted from the given mask
    """

    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    n_objects = len(contours)
    object_areas = [cv2.contourArea(contour) for contour in contours]
    mean_object_area = np.mean(object_areas)
    median_object_area = np.median(object_areas)
    min_object_area = np.min(object_areas)
    max_object_area = np.max(object_areas)

    spatial_properties = {
        'n_objects': n_objects,
        'mean_object_area': float(mean_object_area),
        'median_object_area': int(median_object_area),
        'min_object_area': int(min_object_area),
        'max_object_area': int(max_object_area),
    }

    return spatial_properties


def evaluate_predictions(ground_truth, predictions, threshold):

    """
    Evaluate predictions by comparing them with ground-truth

    Parameters
    ----------
    ground_truth (array-like of shape (height, width)): Ground truth array
    predictions (array-like of shape (height, width)): Predictions array
    threshold (float): Threshold for converting soft predictions into hard labels (0 <= threshold <= 1)

    Returns
    -------
    evaluation_summary (dict): Dictionary of evaluation summary
    """

    dice_coefficient = metrics.binary_dice_coefficient(ground_truth=ground_truth, predictions=predictions, threshold=threshold)
    intersection_over_union = metrics.binary_intersection_over_union(ground_truth=ground_truth, predictions=predictions, threshold=threshold)

    evaluation_summary = {
        'scores': {
            'dice_coefficient': dice_coefficient,
            'intersection_over_union': intersection_over_union
        },
        'statistics':  {
            'ground_truth': {
                'mean': float(np.mean(ground_truth)),
                'sum': int(np.sum(ground_truth))
            },
            'predictions': {
                'mean': float(np.mean(predictions)),
                'sum': int(np.sum(predictions))
            }
        },
        'spatial_properties': {
            'ground_truth': extract_spatial_properties(ground_truth),
            'predictions': extract_spatial_properties(predictions)
        }
    }

    return evaluation_summary
