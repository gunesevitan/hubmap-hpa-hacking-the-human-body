import torch.nn as nn
import segmentation_models_pytorch as smp

import daformer


class UNetDAFormerModel(nn.Module):

    def __init__(self, encoder_class, encoder_args, decoder_args, head_args):

        super(UNetDAFormerModel, self).__init__()

        self.encoder = getattr(smp, encoder_class)(**encoder_args).encoder
        self.decoder = daformer.DAFormerDecoder(**decoder_args)
        self.conv_head = nn.Sequential(
            nn.Conv2d(
                in_channels=self.decoder.decoder_dim,
                out_channels=head_args['classes'],
                kernel_size=1
            ),
            nn.Upsample(
                size=head_args['upsample_size'],
                mode=head_args['upsample_mode'],
                align_corners=head_args['upsample_align_corners']
            ),
        )

    def forward(self, x):

        x = self.encoder(x)
        x, _ = self.decoder(x[1:])
        out = self.conv_head(x)

        return out
