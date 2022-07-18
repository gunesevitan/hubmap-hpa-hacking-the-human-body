import torch.nn as nn
import monai.networks.nets


class MONAISegmentationModel(nn.Module):

    def __init__(self, model_class, model_args):

        super(MONAISegmentationModel, self).__init__()

        self.model = getattr(monai.networks.nets, model_class)(**model_args)

    def forward(self, x):

        return self.model(x)
