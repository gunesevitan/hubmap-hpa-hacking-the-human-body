import json
import numpy as np
import tifffile
import torch
from torch.utils.data import Dataset

import annotation_utils


class SemanticSegmentationDataset(Dataset):

    def __init__(self, image_paths, masks=None, transforms=None, mask_format='rle'):

        self.image_paths = image_paths
        self.masks = masks
        self.transforms = transforms
        self.mask_format = mask_format

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):

        """
        Get the idxth element in the dataset

        Parameters
        ----------
        idx (int): Index of the sample (0 <= idx < len(self.image_paths))

        Returns
        -------
        image (torch.FloatTensor of shape (3, height, width)): Image tensor
        mask (torch.FloatTensor of shape (1, height, width)): Mask tensor
        """

        image = tifffile.imread(str(self.image_paths[idx]))

        if self.masks is not None:

            if self.mask_format == 'rle':
                # Decode RLE mask string into 2d binary semantic segmentation mask array
                mask = annotation_utils.decode_rle_mask(rle_mask=self.masks[idx], shape=image.shape[:2]).T
            elif self.mask_format == 'polygon':
                # Read polygon JSON file and convert it into 2d binary semantic segmentation mask array
                with open(self.masks[idx], mode='r') as f:
                    polygons = json.load(f)
                mask = annotation_utils.polygon_to_mask(polygons=polygons, shape=image.shape[:2])
            else:
                raise ValueError('Invalid mask format')

            if self.transforms is not None:
                # Apply transforms to image and semantic segmentation mask
                transformed = self.transforms(image=image, mask=mask)
                image = transformed['image'].float()
                mask = transformed['mask'].float()
                mask = torch.unsqueeze(mask, dim=0)

            else:
                image = torch.as_tensor(image, dtype=torch.float)
                mask = torch.as_tensor(mask, dtype=torch.float)
                mask = torch.unsqueeze(mask, dim=0)
                # Scale pixel values by max 8 bit pixel value
                image /= 255.

            return image, mask

        else:

            if self.transforms is not None:
                # Apply transforms to image
                transformed = self.transforms(image=image)
                image = transformed['image'].float()
            else:
                image = torch.as_tensor(image, dtype=torch.float)
                # Scale pixel values by max 8 bit pixel value
                image /= 255.

            return image
