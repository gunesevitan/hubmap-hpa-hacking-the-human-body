import torch.nn as nn
import segmentation_models_pytorch as smp
import monai.networks.nets
import transformers


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


class HuggingFaceTransformersModel(nn.Module):

    def __init__(self, model_class, model_args, upsample_args):

        super(HuggingFaceTransformersModel, self).__init__()

        self.model = getattr(transformers, model_class).from_pretrained(**model_args)
        self.upsample = nn.Upsample(
            size=upsample_args['upsample_size'],
            mode=upsample_args['upsample_mode'],
            align_corners=upsample_args['upsample_align_corners']
        )

    def forward(self, x):

        return self.upsample(self.model(x)[0])
