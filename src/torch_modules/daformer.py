import torch
import torch.nn as nn
import torch.nn.functional as F


class MixUpSample(nn.Module):

    def __init__(self, scale_factor=2):

        super(MixUpSample, self).__init__()
        assert (scale_factor != 1)

        self.mixing = nn.Parameter(torch.tensor(0.5))
        self.scale_factor = scale_factor

    def forward(self, x):
        x = self.mixing * F.interpolate(x, scale_factor=self.scale_factor, mode='bilinear', align_corners=False) \
            + (1 - self.mixing) * F.interpolate(x, scale_factor=self.scale_factor, mode='nearest')
        return x


def Conv2dBnReLU(in_channel, out_channel, kernel_size=3, padding=1, stride=1, dilation=1):
    return nn.Sequential(
        nn.Conv2d(in_channel, out_channel, kernel_size=kernel_size, padding=padding, stride=stride, dilation=dilation, bias=False),
        nn.BatchNorm2d(out_channel),
        nn.ReLU(inplace=True),
    )


class ASPP(nn.Module):

    def __init__(self, in_channel, channel, dilation):

        super(ASPP, self).__init__()

        self.conv = nn.ModuleList()
        for d in dilation:
            self.conv.append(
                Conv2dBnReLU(
                    in_channel,
                    channel,
                    kernel_size=1 if d == 1 else 3,
                    dilation=d,
                    padding=0 if d == 1 else d,
                )
            )

        self.out = Conv2dBnReLU(
            len(dilation) * channel,
            channel,
            kernel_size=3,
            padding=1,
        )

    def forward(self, x):
        aspp = []
        for conv in self.conv:
            aspp.append(conv(x))
        aspp = torch.cat(aspp, dim=1)
        out = self.out(aspp)
        return out


class DSConv2d(nn.Module):

    def __init__(self, in_channel, out_channel, kernel_size, stride=1, padding=0, dilation=1):

        super(DSConv2d, self).__init__()

        self.depthwise = nn.Sequential(
            nn.Conv2d(in_channel, in_channel, kernel_size, stride=stride, padding=padding, dilation=dilation),
            nn.BatchNorm2d(in_channel),
            nn.ReLU(inplace=True)
        )

        self.pointwise = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x


class DSASPP(nn.Module):

    def __init__(self, in_channel, channel, dilation):

        super(DSASPP, self).__init__()

        self.conv = nn.ModuleList()
        for d in dilation:
            if d == 1:
                self.conv.append(
                    Conv2dBnReLU(
                        in_channel,
                        channel,
                        kernel_size=1 if d == 1 else 3,
                        dilation=d,
                        padding=0 if d == 1 else d,
                    )
                )
            else:
                self.conv.append(
                    DSConv2d(
                        in_channel,
                        channel,
                        kernel_size=3,
                        dilation=d,
                        padding=d,
                    )
                )

        self.out = Conv2dBnReLU(
            len(dilation) * channel,
            channel,
            kernel_size=3,
            padding=1,
        )

    def forward(self, x):
        aspp = []
        for conv in self.conv:
            aspp.append(conv(x))
        aspp = torch.cat(aspp, dim=1)
        out = self.out(aspp)
        return out


class DAFormerDecoder(nn.Module):

    def __init__(self, encoder_dim=(32, 64, 160, 256), decoder_dim=256, dilation=(1, 6, 12, 18), use_bn_mlp=True, fuse='conv3x3'):

        super(DAFormerDecoder, self).__init__()

        self.decoder_dim = decoder_dim

        self.mlp = nn.ModuleList([
            nn.Sequential(
                *(
                    (nn.Conv2d(dim, decoder_dim, 1, padding=0, bias=False),
                     nn.BatchNorm2d(decoder_dim),
                     nn.ReLU(inplace=True),
                     ) if use_bn_mlp else
                    (nn.Conv2d(dim, decoder_dim, 1, padding=0, bias=True),)
                ),
                MixUpSample(2 ** i) if i != 0 else nn.Identity(),
            ) for i, dim in enumerate(encoder_dim)])

        if fuse == 'conv1x1':
            self.fuse = nn.Sequential(
                nn.Conv2d(len(encoder_dim) * decoder_dim, decoder_dim, 1, padding=0, bias=False),
                nn.BatchNorm2d(decoder_dim),
                nn.ReLU(inplace=True),
            )

        if fuse == 'conv3x3':
            self.fuse = nn.Sequential(
                nn.Conv2d(len(encoder_dim) * decoder_dim, decoder_dim, 3, padding=1, bias=False),
                nn.BatchNorm2d(decoder_dim),
                nn.ReLU(inplace=True),
            )

        if fuse == 'aspp':
            self.fuse = ASPP(
                decoder_dim * len(encoder_dim),
                decoder_dim,
                dilation,
            )

        if fuse == 'ds-aspp':
            self.fuse = DSASPP(
                decoder_dim * len(encoder_dim),
                decoder_dim,
                dilation,
            )

    def forward(self, feature):
        out = []
        for i, f in enumerate(feature):
            f = self.mlp[i](f)
            out.append(f)
        x = self.fuse(torch.cat(out, dim=1))
        return x, out
