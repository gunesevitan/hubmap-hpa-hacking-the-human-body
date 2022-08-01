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
        A.Resize(
            height=transform_parameters['resize_height'],
            width=transform_parameters['resize_width'],
            interpolation=cv2.INTER_NEAREST,
            always_apply=True
        ),
        A.HorizontalFlip(p=transform_parameters['horizontal_flip_probability']),
        A.VerticalFlip(p=transform_parameters['vertical_flip_probability']),
        A.RandomRotate90(p=transform_parameters['random_rotate_90_probability']),
        A.ShiftScaleRotate(
            shift_limit=transform_parameters['shift_limit'],
            scale_limit=transform_parameters['scale_limit'],
            rotate_limit=transform_parameters['rotate_limit'],
            p=transform_parameters['shift_scale_rotate_probability']
        ),
        A.HueSaturationValue(
            hue_shift_limit=transform_parameters['hue_shift_limit'],
            sat_shift_limit=transform_parameters['saturation_shift_limit'],
            val_shift_limit=transform_parameters['value_shift_limit'],
            p=transform_parameters['hue_saturation_value_probability']
        ),
        A.RandomBrightnessContrast(
            brightness_limit=transform_parameters['brightness_limit'],
            contrast_limit=transform_parameters['contrast_limit'],
            p=transform_parameters['random_brightness_contrast_probability']
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
                border_mode=cv2.BORDER_REPLICATE,
                p=transform_parameters['grid_distortion_probability']
            ),
            A.OpticalDistortion(
                distort_limit=transform_parameters['optical_distortion_distort_limit'],
                shift_limit=transform_parameters['optical_distortion_shift_limit'],
                interpolation=cv2.INTER_NEAREST,
                border_mode=cv2.BORDER_REPLICATE,
                p=transform_parameters['optical_distortion_probability']
            )
        ], p=transform_parameters['distortion_probability']),
        A.OneOf([
            A.ChannelShuffle(p=transform_parameters['channel_shuffle_probability']),
            A.ChannelDropout(
                channel_drop_range=transform_parameters['channel_dropout_channel_drop_range'],
                fill_value=transform_parameters['channel_dropout_fill_value'],
                p=transform_parameters['channel_dropout_probability']
            )
        ], p=transform_parameters['channel_transform_probability']),
        A.OneOf([
            A.CoarseDropout(
                max_holes=transform_parameters['coarse_dropout_max_holes'],
                max_height=transform_parameters['coarse_dropout_max_height'],
                max_width=transform_parameters['coarse_dropout_max_width'],
                min_holes=transform_parameters['coarse_dropout_min_holes'],
                min_height=transform_parameters['coarse_dropout_min_height'],
                min_width=transform_parameters['coarse_dropout_min_width'],
                fill_value=transform_parameters['coarse_dropout_fill_value'],
                mask_fill_value=transform_parameters['coarse_dropout_mask_fill_value'],
                p=transform_parameters['coarse_dropout_probability']
            ),
            A.PixelDropout(
                dropout_prob=transform_parameters['pixel_dropout_dropout_probability'],
                per_channel=transform_parameters['pixel_dropout_per_channel'],
                drop_value=transform_parameters['pixel_dropout_drop_value'],
                mask_drop_value=transform_parameters['pixel_dropout_mask_drop_value'],
                p=transform_parameters['pixel_dropout_probability']
            ),
            A.MaskDropout(
                max_objects=transform_parameters['mask_dropout_max_objects'],
                image_fill_value=transform_parameters['mask_dropout_image_fill_value'],
                mask_fill_value=transform_parameters['mask_dropout_mask_fill_value'],
                p=transform_parameters['mask_dropout_probability']
            )
        ], p=transform_parameters['dropout_probability']),
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
