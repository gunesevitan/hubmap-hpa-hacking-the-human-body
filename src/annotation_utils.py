import json
import numpy as np
import cv2

import settings


def decode_rle_mask(rle_mask, shape):

    """
    Decode run-length encoded segmentation mask string into 2d array

    Parameters
    ----------
    rle_mask (str): Run-length encoded segmentation mask string
    shape (tuple of shape (2)): Height and width of the mask

    Returns
    -------
    mask (numpy.ndarray of shape (height, width)): Decoded 2d segmentation mask
    """

    rle_mask = rle_mask.split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (rle_mask[0:][::2], rle_mask[1:][::2])]
    starts -= 1
    ends = starts + lengths

    mask = np.zeros((shape[0] * shape[1]), dtype=np.uint8)
    for start, end in zip(starts, ends):
        mask[start:end] = 1

    mask = mask.reshape(shape[0], shape[1])
    return mask


def encode_rle_mask(mask):

    """
    Encode 2d array into run-length encoded segmentation mask string

    Parameters
    ----------
    mask (numpy.ndarray of shape (height, width)): 2d segmentation mask

    Returns
    -------
    rle_mask (str): Run-length encoded segmentation mask string
    """

    mask = mask.flatten()
    mask = np.concatenate([[0], mask, [0]])
    runs = np.where(mask[1:] != mask[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return ' '.join(str(x) for x in runs)


def get_segmentation_mask_from_df(df, image_id):

    """
    Retrieve binary segmentation mask of given image_id from given dataframe

    Parameters
    ----------
    df (pandas.DataFrame of shape (n_rows, n_columns)): Dataframe with id, rle, img_height and img_width columns
    image_id (str): ID of the image

    Returns
    -------
    segmentation_mask (numpy.ndarray of shape (img_height, img_width)): Binary segmentation mask
    """

    rle_masks = df.loc[df['id'] == image_id, 'rle']
    if rle_masks.shape[0] > 0:
        segmentation_mask = decode_rle_mask(
            rle_masks.values[0],
            shape=(
                int(df.loc[df['id'] == image_id, 'img_height']),
                int(df.loc[df['id'] == image_id, 'img_width'])
            )
        )
    else:
        raise KeyError(f'Image ID {image_id} is not found in the given dataframe')

    return segmentation_mask


def get_segmentation_polygons_from_json(annotation_directory, image_id):

    """
    Retrieve polygons of given image_id from given directory

    Parameters
    ----------
    annotation_directory (str): Directory of the annotations relative to root/data/
    image_id (str): ID of the image

    Returns
    -------
    polygons (list of shape (n_polygons, n_points, 2)): Polygons
    """

    with open(settings.DATA / annotation_directory / f'{image_id}.json', mode='r') as f:
        polygons = json.load(f)

    return polygons


def polygon_to_mask(polygons, shape):

    """
    Create binary segmentation mask from polygons

    Parameters
    ----------
    polygons (list of shape (n_polygons, n_points, 2)): Polygons
    shape (tuple of shape (2)): Height and width of the mask

    Returns
    -------
    segmentation_mask (numpy.ndarray of shape (height, width)): 2d segmentation mask
    """

    segmentation_mask = np.zeros(shape)

    for polygon in polygons:
        # Convert list of points to tuple pairs of X and Y coordinates
        points = np.array(polygon).reshape(-1, 2)
        # Draw mask from the polygon
        cv2.fillPoly(segmentation_mask, [points], 1, lineType=cv2.LINE_8, shift=0)

    segmentation_mask = np.array(segmentation_mask).astype(np.uint8)

    return segmentation_mask
