import cv2
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2


def get_semantic_segmentation_transforms(**transform_parameters):

    """
    Get transforms for semantic segmentation dataset

    Parameters
    ----------
    transform_parameters (dict): Dictionary of transform parameters

    Returns
    -------
    transforms (dict): Transforms for training, validation and test sets
    """

    train_transforms = A.Compose([
        A.Resize(
            height=transform_parameters['resize_height'],
            width=transform_parameters['resize_width'],
            interpolation=cv2.INTER_LANCZOS4,
            always_apply=True
        ),
        A.HorizontalFlip(p=transform_parameters['horizontal_flip_probability']),
        A.VerticalFlip(p=transform_parameters['vertical_flip_probability']),
        A.Normalize(
            mean=transform_parameters['normalize_mean'],
            std=transform_parameters['normalize_std'],
            max_pixel_value=transform_parameters['normalize_max_pixel_value'],
            always_apply=True
        ),
        ToTensorV2(always_apply=True)
    ])

    val_transforms = A.Compose([
        A.Resize(
            height=transform_parameters['resize_height'],
            width=transform_parameters['resize_width'],
            interpolation=cv2.INTER_LANCZOS4,
            always_apply=True
        ),
        A.Normalize(
            mean=transform_parameters['normalize_mean'],
            std=transform_parameters['normalize_std'],
            max_pixel_value=transform_parameters['normalize_max_pixel_value'],
            always_apply=True
        ),
        ToTensorV2(always_apply=True)
    ])

    transforms = {'train': train_transforms, 'val': val_transforms}
    return transforms
