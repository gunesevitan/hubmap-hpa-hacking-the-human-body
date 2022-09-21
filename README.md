# HuBMAP + HPA - Hacking the Human Body

## Installation

```
git clone https://github.com/gunesevitan/hubmap-hpa-hacking-the-human-body.git
cd hubmap-hpa-hacking-the-human-body
virtualenv --python=/usr/bin/python3.10 venv
source venv/bin/activate
pip install -r requirements.txt
```

You can install internal dataset either manually from https://www.kaggle.com/competitions/hubmap-organ-segmentation/data or using Kaggle API.
Ensure kaggle.json is in the location `~/.kaggle/kaggle.json` to use the API

```
pip install kaggle
cd data
kaggle competitions download hubmap-organ-segmentation
unzip hubmap-organ-segmentation.zip
```

External datasets can be downloaded from their original sources. They aren't provided in this repository because of the license issues.

## Project Structure

```
hubmap-hpa-hacking-the-human-body/
├─ data/
├─ eda/
├─ logs/
├─ models/
│  ├─ coat/
│  ├─ coat_daformer_semantic_segmentation_raw_coat_lite_medium_768/
│  ├─ coat_daformer_semantic_segmentation_raw_coat_lite_small_1088/
│  ├─ segformer_semantic_segmentation_raw_mitb3_768
│  ├─ unet_semantic_segmentation_pretraining
│  ├─ unet_semantic_segmentation_raw_efficientnetb2_1184
│  ├─ unet_semantic_segmentation_raw_efficientnetb3_1088
│  ├─ unet_semantic_segmentation_raw_efficientnetb4_992
│  ├─ unet_semantic_segmentation_raw_efficientnetb5_864
│  ├─ unet_semantic_segmentation_raw_efficientnetb6_768
│  ├─ unet_semantic_segmentation_raw_efficientnetb7_672
├─ notebook/
├─ resources/
├─ src/
│  ├─ external_data/
│  ├─ torch_modules/
│  ├─ utilities/
├─ venv/
├─ .gitattributes
├─ .gitignore
├─ requirements.txt
├─ README.md
```

Internal and external datasets are inside the data directory.

Data analysis and visualizations are inside the eda directory.

Logs are inside the logs directory

Pre-trained model weights and model configurations are inside the models directory.

Notebooks are inside the notebook directory.

Papers and other resources are inside the resources directory.

Python modules are inside src directory.

## Software and Hardware

```
Python 3.10.4
QuPath 0.3.2
ImageJ 1.53
Ubuntu 22.04.01 LTS
CPU: AMD Ryzen 9 5950X 16-Core Processor
GPU: NVIDIA GeForce RTX 3090
```

## Dataset

* Raw HPA images (pseudo labels on kidney and large intestine images)
* Raw Single HuBMAP test image (pseudo labels on single spleen image)
* Raw HuBMAP Colonic Crypt Dataset
* GTEx WSI Slices (pseudo labels on spleen and prostate images) 
  * Samples with no abnormalities are selected and downloaded from the platform
  * Slices are extracted using QuPath with appropriate pixel size and saved as jpeg files using ImageJ
  * Pseudo labels are generated on extracted slices

## Models

* CoaT lite-medium encoder (ImageNet pretrained) and DAFormer decoder trained on raw images of size 768
* CoaT lite-small encoder (ImageNet pretrained) and DAFormer decoder trained on raw images of size 1088
* SegFormer MiT-B3 (ADE20k pretrained) trained on raw images of size 768
* UNet EfficientNet-B2 encoder (ImageNet pretrained) trained on raw images of size 1184
* UNet EfficientNet-B3 encoder (ImageNet pretrained) trained on raw images of size 1088
* UNet EfficientNet-B4 encoder (ImageNet pretrained) trained on raw images of size 992
* UNet EfficientNet-B5 encoder (ImageNet pretrained) trained on raw images of size 864
* UNet EfficientNet-B6 encoder (ImageNet pretrained) trained on raw images of size 768
* UNet EfficientNet-B7 encoder (ImageNet pretrained) trained on raw images of size 672

## Validation

5 folds of cross-validation is used as the validation scheme. Splits are stratified on organ type and dataset is shuffled before splitting.

## Training
 
**Loss Function**: Weighted loss function (0.5 x Binary Cross Entropy with Logits Loss + 0.5 x Tversky Loss)

**Optimizer**: AdamW with 2e-4 initial learning rate and learning rate is multiplied with 0.75 after every 1000 steps

**Batch Size**: Training: 4 - Validation: 8

**Stochastic Weight Averaging**: SWA is triggered after 30 epochs with 1e-5 learning rate

**Early Stopping**: Early Stopping is triggered after 15 epochs with no validation loss improvement

## Training Augmentations

* Resize
* Horizontal flip by 50% chance
* Vertical flip by 50% chance
* Random 90-degree rotation by 25% chance
* Random shift-scale-rotate by 50% chance (only positive scale)
* Random hue-saturation-value by 50% chance (higher saturation)
* Random brightness-contrast by 25% chance
* CLAHE or histogram equalization by 20% chance
* Grid distortion or optical distortion by 25% chance
* Coarse dropout or pixel dropout or mask dropout by 10% chance
* Normalize by dataset statistics

## Test-time Augmentations

* Horizontal flip
* Vertical flip
* Horizontal flip + vertical flip
* Stain normalization
* Stain normalization + horizontal flip
* Stain normalization + vertical flip
* Stain normalization + horizontal flip + vertical flip

Stain normalization is applied on only HuBMAP images.
A random HPA image with same organ type is selected as the domain image and target images' stain normalized to that.

## Ensemble

5 models are used in the ensemble.
Models trained on larger images (1088) performed better on kidney and large intestine images.
Models trained on smaller images (768) performed better on prostate and spleen images.
None of the models did good on lung images because of the noisy labels, but EfficientNet-B6 was the best among them.
Sigmoid function is applied to logits at this stage because model predictions were on different scale.

* CoaT lite-medium encoder (ImageNet pretrained) and DAFormer decoder trained on raw images of size 768
* CoaT lite-small encoder (ImageNet pretrained) and DAFormer decoder trained on raw images of size 1088
* SegFormer MiT-B3 (ADE20k pretrained) trained on raw images of size 768
* UNet EfficientNet-B3 encoder (ImageNet pretrained) trained on raw images of size 1088
* UNet EfficientNet-B6 encoder (ImageNet pretrained) trained on raw images of size 768

Different ensemble weights are used for each organ type and those weights are found by trial and error.

**Kidney**:
  * UNet EfficientNet-B3: 0.20
  * UNet EfficientNet-B6: 0.25
  * SegFormer MiT-B3: 0.10
  * CoaT lite small: 0.20
  * CoaT lite medium: 0.25

**Prostate**:
  * UNet EfficientNet-B3: 0.20
  * UNet EfficientNet-B6: 0.25
  * SegFormer MiT-B3: 0.15
  * CoaT lite small: 0.15
  * CoaT lite medium: 0.25

**Spleen**:
  * UNet EfficientNet-B3: 0.15
  * UNet EfficientNet-B6: 0.25
  * SegFormer MiT-B3: 0.15
  * CoaT lite small: 0.20
  * CoaT lite medium: 0.25

**Large Intestine**:
  * UNet EfficientNet-B3: 0.20
  * UNet EfficientNet-B6: 0.25
  * SegFormer MiT-B3: 0.10
  * CoaT lite small: 0.20
  * CoaT lite medium: 0.25

**Lung**:
  * UNet EfficientNet-B3: 0.20
  * UNet EfficientNet-B6: 0.70
  * SegFormer MiT-B3: 0.0
  * CoaT lite small: 0.0
  * CoaT lite medium: 0.10

## Post-processing

Hard labels are obtained using different thresholds for each organ type. Thresholds are found by trial and error.

* **Kidney**: 0.25
* **Prostate**: 0.20
* **Spleen**: 0.25
* **Large Intestine**: 0.20
* **Lung**: 0.05
