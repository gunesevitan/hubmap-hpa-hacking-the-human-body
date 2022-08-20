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

    def __init__(self, model_class, model_args):

        super(HuggingFaceTransformersModel, self).__init__()

        self.model = getattr(transformers, model_class).from_pretrained(model_args['model_weights'])
        self.model.decode_head.dropout = nn.Dropout(model_args['decoder_dropout_rate'])
        self.model.decode_head.classifier = nn.Conv2d(256, 1, kernel_size=(1, 1), stride=(1, 1))
        self.upsample = nn.Upsample(
            size=model_args['upsample_size'],
            mode=model_args['upsample_mode'],
            align_corners=model_args['upsample_align_corners']
        )

    def forward(self, x):

        return self.upsample(self.model(x)[0])
