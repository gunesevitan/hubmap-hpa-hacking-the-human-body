import numpy as np


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
