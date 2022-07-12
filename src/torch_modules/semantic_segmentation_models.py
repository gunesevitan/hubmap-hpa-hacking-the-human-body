import torch.nn as nn
import segmentation_models_pytorch as smp


class SemanticSegmentationModel(nn.Module):

    def __init__(self, model_class, model_args):

        super(SemanticSegmentationModel, self).__init__()

        self.model = getattr(smp, model_class)(**model_args)

    def forward(self, x):

        return self.model(x)
