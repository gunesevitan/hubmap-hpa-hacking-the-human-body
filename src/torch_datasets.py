import json
import cv2
import tifffile
import torch
from torch.utils.data import Dataset

import annotation_utils
import preprocessing


class SemanticSegmentationDataset(Dataset):

    def __init__(self, image_paths, masks=None, data_sources=None, transforms=None, mask_format='rle', crop_black_border=False, crop_background=False):

        self.image_paths = image_paths
        self.masks = masks
        self.data_sources = data_sources
        self.transforms = transforms
        self.mask_format = mask_format
        self.crop_black_border = crop_black_border
        self.crop_background = crop_background

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

        image_format = self.image_paths[idx].split('/')[-1].split('.')[-1]
        if image_format == 'tiff':
            image = tifffile.imread(str(self.image_paths[idx]))
        elif image_format == 'png':
            image = cv2.imread(str(self.image_paths[idx]))
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            raise ValueError(f'Invalid image format: {image_format}')

        if self.masks is not None:

            if self.mask_format == 'rle':
                # Decode RLE mask string into 2d binary semantic segmentation mask array
                mask = annotation_utils.decode_rle_mask(rle_mask=self.masks[idx], shape=image.shape[:2])
                data_source = self.data_sources[idx]
                # Transpose raw HPA and HuBMAP masks
                if data_source == 'Hubmap' or data_source == 'HPA':
                    mask = mask.T
            elif self.mask_format == 'polygon':
                # Read polygon JSON file and convert it into 2d binary semantic segmentation mask array
                with open(self.masks[idx], mode='r') as f:
                    polygons = json.load(f)
                mask = annotation_utils.polygon_to_mask(polygons=polygons, shape=image.shape[:2])
            else:
                raise ValueError(f'Invalid mask format: {self.mask_format}')

            if self.crop_black_border or self.crop_background:
                # Crop black border or background from the image and mask
                image, mask = preprocessing.crop_image(
                    image=image,
                    mask=mask,
                    crop_black_border=self.crop_black_border,
                    crop_background=self.crop_background
                )

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

            if self.crop_black_border or self.crop_background:
                # Crop black border or background from the image
                image, mask = preprocessing.crop_image(
                    image=image,
                    mask=None,
                    crop_black_border=self.crop_black_border,
                    crop_background=self.crop_background
                )

            if self.transforms is not None:
                # Apply transforms to image
                transformed = self.transforms(image=image)
                image = transformed['image'].float()
            else:
                image = torch.as_tensor(image, dtype=torch.float)
                # Scale pixel values by max 8 bit pixel value
                image /= 255.

            return image
