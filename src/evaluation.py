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


def evaluate_predictions(ground_truth, predictions, thresholds=(0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7)):

    """
    Evaluate predictions and ground-truth if it is given

    Parameters
    ----------
    ground_truth (array-like of shape (height, width)): Ground truth array
    predictions (array-like of shape (height, width)): Predictions array
    thresholds (tuple of shape (n_thresholds)): Thresholds for converting soft predictions into hard labels (0 <= threshold <= 1)

    Returns
    -------
    evaluation_summary (dict): Dictionary of evaluation summary
    """

    if ground_truth is not None:
        dice_coefficients = metrics.mean_binary_dice_coefficient(ground_truth=ground_truth, predictions=predictions, thresholds=thresholds)
        intersection_over_unions = metrics.mean_binary_intersection_over_union(ground_truth=ground_truth, predictions=predictions, thresholds=thresholds)
        scores = {
            'dice_coefficients': dice_coefficients[0],
            'mean_dice_coefficient': dice_coefficients[1],
            'intersection_over_unions': intersection_over_unions[0],
            'mean_intersection_over_union': intersection_over_unions[1],
        }
    else:
        scores = None

    evaluation_summary = {
        'scores': scores,
        'statistics':  {
            'ground_truth': {
                'mean': float(np.mean(ground_truth)) if ground_truth is not None else None,
                'sum': int(np.sum(ground_truth)) if ground_truth is not None else None
            },
            'predictions': {
                'mean': float(np.mean(predictions)),
                'sum': int(np.sum(predictions))
            }
        },
        'spatial_properties': {
            'ground_truth': extract_spatial_properties(ground_truth) if ground_truth is not None else None,
            'predictions': extract_spatial_properties(np.uint8(predictions >= np.median(thresholds)))
        }
    }

    return evaluation_summary
