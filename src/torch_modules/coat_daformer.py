import torch
import torch.nn as nn

from . import coat
from . import daformer


class CoaTDAFormer(nn.Module):

    def __init__(self, encoder_name, decoder_args, encoder_weights=None):

        super(CoaTDAFormer, self).__init__()

        self.encoder = getattr(coat, encoder_name)()
        if encoder_weights is not None:
            if encoder_weights == 'coat_lite_medium':
                state_dict = torch.load(encoder_weights)['model']
            else:
                state_dict = torch.load(encoder_weights)
            self.encoder.load_state_dict(
                state_dict=state_dict,
                strict=False
            )
        self.decoder = daformer.DAFormerDecoder(**decoder_args)
        self.conv_head = nn.Sequential(
            nn.Conv2d(self.decoder.decoder_dim, 1, kernel_size=1),
            nn.Upsample(scale_factor=4, mode='bilinear', align_corners=False),
        )

    def forward(self, x):

        x = self.encoder(x)
        last, decoder = self.decoder(x)
        out = self.conv_head(last)

        return out
