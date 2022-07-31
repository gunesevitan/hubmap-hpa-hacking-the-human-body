import numpy as np
import cv2
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2
from albumentations import ImageOnlyTransform


class Scale(ImageOnlyTransform):

    def apply(self, image, **kwargs):

        """
        Scale pixel values between 0 and 1

        Parameters
        ----------
        image (numpy.ndarray of shape (height, width)): Image array

        Returns
        -------
        image (numpy.ndarray of shape (height, width)): Image array divided by max 8-bit integer
        """

        image = np.float32(image) / 255.

        return image


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
        A.RandomResizedCrop(
            height=transform_parameters['resize_height'],
            width=transform_parameters['resize_width'],
            scale=transform_parameters['resize_scale'],
            ratio=transform_parameters['resize_ratio'],
            interpolation=cv2.INTER_NEAREST,
            always_apply=True
        ),
        A.HorizontalFlip(p=transform_parameters['horizontal_flip_probability']),
        A.VerticalFlip(p=transform_parameters['vertical_flip_probability']),
        A.RandomRotate90(p=transform_parameters['random_rotate_90_probability']),
        A.ColorJitter(
            brightness=transform_parameters['brightness'],
            contrast=transform_parameters['contrast'],
            saturation=transform_parameters['saturation'],
            hue=transform_parameters['hue'],
            p=transform_parameters['color_jitter_probability']
        ),
        A.OneOf([
            A.CLAHE(
                clip_limit=transform_parameters['clahe_clip_limit'],
                tile_grid_size=transform_parameters['clahe_tile_grid_size'],
                p=transform_parameters['clahe_probability']
            ),
            A.Equalize(
                mode='cv',
                by_channels=True,
                p=transform_parameters['equalize_probability']
            )
        ], p=transform_parameters['histogram_equalization_probability']),
        A.OneOf([
            A.GridDistortion(
                num_steps=transform_parameters['grid_distortion_num_steps'],
                distort_limit=transform_parameters['grid_distortion_distort_limit'],
                interpolation=cv2.INTER_NEAREST,
                border_mode=cv2.BORDER_CONSTANT,
                value=(0, 0, 0),
                mask_value=(0, 0, 0),
                p=transform_parameters['grid_distortion_probability']
            ),
            A.OpticalDistortion(
                distort_limit=transform_parameters['optical_distortion_distort_limit'],
                shift_limit=transform_parameters['optical_distortion_shift_limit'],
                interpolation=cv2.INTER_NEAREST,
                border_mode=cv2.BORDER_CONSTANT,
                value=(0, 0, 0),
                mask_value=(0, 0, 0),
                p=transform_parameters['optical_distortion_probability']
            )
        ], p=transform_parameters['distortion_probability']),
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
            interpolation=cv2.INTER_NEAREST,
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
