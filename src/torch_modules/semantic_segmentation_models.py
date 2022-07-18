import torch.nn as nn
import segmentation_models_pytorch as smp
import monai.networks.nets


class SemanticSegmentationModel(nn.Module):

    def __init__(self, model_module, model_class, model_args):

        super(SemanticSegmentationModel, self).__init__()

        if model_module == 'smp':
            self.model = getattr(smp, model_class)(**model_args)
        elif model_module == 'monai':
            self.model = getattr(monai.networks.nets, model_class)(**model_args)
        else:
            raise ValueError('Invalid model_module')

    def forward(self, x):

        return self.model(x)
