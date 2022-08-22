import numpy as np
import cv2
import tifffile
import torch
from torch.utils.data import Dataset
import staintools

import annotation_utils


imaging_measurements = {
  'HPA': {
    'pixel_size': {
      'kidney': 0.4,
      'prostate': 0.4,
      'largeintestine': 0.4,
      'spleen': 0.4,
      'lung': 0.4
    },
    'tissue_thickness': {
      'kidney': 4,
      'prostate': 4,
      'largeintestine': 4,
      'spleen': 4,
      'lung': 4
    }
  },
  'Hubmap': {
    'pixel_size': {
      'kidney': 0.5,
      'prostate': 6.263,
      'largeintestine': 0.229,
      'spleen': 0.4945,
      'lung': 0.7562
    },
    'tissue_thickness': {
      'kidney': 10,
      'prostate': 5,
      'largeintestine': 8,
      'spleen': 4,
      'lung': 5
    }
  }
}


class SemanticSegmentationDataset(Dataset):

    def __init__(self, image_paths, organs, data_sources, masks=None,
                 imaging_measurement_adaptation_probability=0, standardize_luminosity_probability=0, transforms=None):

        self.image_paths = image_paths
        self.organs = organs
        self.data_sources = data_sources
        self.masks = masks
        self.imaging_measurement_adaptation_probability = imaging_measurement_adaptation_probability
        self.standardize_luminosity_probability = standardize_luminosity_probability
        self.transforms = transforms

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

        organ = self.organs[idx]
        data_source = self.data_sources[idx]

        if data_source == 'HPA' or data_source == 'Hubmap' or data_source == 'HuBMAP_Colonic_Crypt':
            image = tifffile.imread(str(self.image_paths[idx]))
        elif data_source == 'GTEx':
            image = cv2.imread(str(self.image_paths[idx]))
        else:
            raise ValueError(f'Invalid data source: {data_source}')

        if data_source == 'HPA':
            if self.imaging_measurement_adaptation_probability > 0:
                if np.random.rand() < self.imaging_measurement_adaptation_probability:
                    # Simulate pixel size artifacts in HPA images randomly
                    domain_pixel_size = imaging_measurements['HPA']['pixel_size'][organ]
                    target_pixel_size = imaging_measurements['Hubmap']['pixel_size'][organ]
                    pixel_size_scale_factor = domain_pixel_size / target_pixel_size

                    image_resized = cv2.resize(image, dsize=None, fx=pixel_size_scale_factor, fy=pixel_size_scale_factor, interpolation=cv2.INTER_LINEAR)
                    image = cv2.resize(image_resized, dsize=(image.shape[1], image.shape[0]), interpolation=cv2.INTER_LINEAR)

        if self.standardize_luminosity_probability > 0:
            if np.random.rand() < self.standardize_luminosity_probability:
                # Standardize luminosity
                image = staintools.LuminosityStandardizer.standardize(image)

        if self.masks is not None:

            # Decode RLE mask string into 2d binary semantic segmentation mask array
            mask = annotation_utils.decode_rle_mask(rle_mask=self.masks[idx], shape=image.shape[:2])
            if data_source == 'Hubmap' or data_source == 'HPA':
                mask = mask.T

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
