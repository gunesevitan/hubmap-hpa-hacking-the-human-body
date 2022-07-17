import logging
import numpy as np
import cv2


def crop_image(image, mask=None, crop_black_border=True, crop_background=True, verbose=False):

    """
    Crop borders and background from the image

    Parameters
    ----------
    image (numpy.ndarray of shape (img_height, img_width, 3)): Image array
    mask (numpy.ndarray of shape (img_height, img_width)): Mask array
    crop_black_border (bool): Whether to crop black border or not
    crop_background (bool): Whether to crop background or not
    verbose (bool): Verbosity switch

    Returns
    -------
    image (numpy.ndarray of shape (img_height, img_width, 3)): Image array
    mask (numpy.ndarray of shape (img_height, img_width)): Mask array
    """

    if crop_black_border:
        black_horizontal_lines = np.all(image == 0, axis=(1, 2))
        image = image[~black_horizontal_lines, :, :]
        if mask is not None:
            mask = mask[~black_horizontal_lines, :]
        if verbose:
            logging.info(f'Image Shape: {image.shape} cropped {np.sum(black_horizontal_lines)} horizontal lines')
        black_vertical_lines = np.all(image == 0, axis=(0, 2))
        image = image[:, ~black_vertical_lines, :]
        if mask is not None:
            mask = mask[:, ~black_vertical_lines]
        if verbose:
            logging.info(f'Image Shape: {image.shape} cropped {np.sum(black_vertical_lines)} vertical lines')

    if crop_background:
        grayscale_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        _, thresholded_image = cv2.threshold(grayscale_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        white_horizontal_lines = np.mean(thresholded_image, axis=1) >= 254
        thresholded_image = thresholded_image[~white_horizontal_lines, :]
        image = image[~white_horizontal_lines, :, :]
        if mask is not None:
            mask = mask[~white_horizontal_lines, :]
        if verbose:
            logging.info(f'Image Shape: {image.shape} cropped {np.sum(white_horizontal_lines)} horizontal lines')

        white_vertical_lines = np.mean(thresholded_image, axis=0) >= 254
        image = image[:, ~white_vertical_lines, :]
        if mask is not None:
            mask = mask[:, ~white_vertical_lines]
        if verbose:
            logging.info(f'Image Shape: {image.shape} cropped {np.sum(white_vertical_lines)} horizontal lines')

    return image, mask
