import torch
import torch.nn as nn

import coat
import daformer


class CoaTDAFormer(nn.Module):

    def __init__(self, encoder_name, decoder_args, encoder_weights=None):

        super(CoaTDAFormer, self).__init__()

        self.encoder = getattr(coat, encoder_name)()
        if encoder_weights is not None:
            self.encoder.load_state_dict(
                state_dict=torch.load(encoder_weights)['model'],
                strict=False
            )
        self.decoder = daformer.DAFormerDecoder(**decoder_args)
        self.conv_head = nn.Sequential(
            nn.Conv2d(self.decoder.decoder_dim, 1, kernel_size=1),
            nn.Upsample(scale_factor=4, mode='bilinear', align_corners=False),
        )

    def forward(self, x):

        x = self.encoder(x)
        for a in x:
            print(a.shape)

        last, decoder = self.decoder(x)
        out = self.conv_head(last)

        return out


if __name__ == '__main__':

    m = CoaTDAFormer(
        encoder_name='coat_lite_medium',
        decoder_args={
            'encoder_dim': (128, 256, 320, 512),
            'decoder_dim': 256,
            'dilation': (1, 6, 12, 18),
            'use_bn_mlp': True,
            'fuse': 'conv3x3'
        },
        encoder_weights='../../models/coat/coat_lite_medium.pth',

    )

    x = torch.rand(1, 3, 224, 224)