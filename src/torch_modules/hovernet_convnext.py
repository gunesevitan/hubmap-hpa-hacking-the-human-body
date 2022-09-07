import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import DropPath
from collections import OrderedDict


def crop_op(x, cropping, data_format='NCHW'):

    crop_t = cropping[0] // 2
    crop_b = cropping[0] - crop_t
    crop_l = cropping[1] // 2
    crop_r = cropping[1] - crop_l

    if data_format == 'NCHW':
        x = x[:, :, crop_t:-crop_b, crop_l:-crop_r]
    else:
        x = x[:, crop_t:-crop_b, crop_l:-crop_r, :]
    return x


def crop_to_shape(x, y, data_format='NCHW'):

    assert (
            y.shape[0] <= x.shape[0] and y.shape[1] <= x.shape[1]
    ), 'Ensure that y dimensions are smaller than x dimensions!'

    x_shape = x.size()
    y_shape = y.size()

    if data_format == 'NCHW':
        crop_shape = (x_shape[2] - y_shape[2], x_shape[3] - y_shape[3])
    else:
        crop_shape = (x_shape[1] - y_shape[1], x_shape[2] - y_shape[2])

    return crop_op(x, crop_shape, data_format)


class Net(nn.Module):

    def weights_init(self):

        for m in self.modules():
            classname = m.__class__.__name__
            if 'linear' in classname.lower():
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

    def forward(self, x):
        return x


class LayerNorm(nn.Module):

    def __init__(self, normalized_shape, eps=1e-6, data_format='channels_last'):

        super(LayerNorm, self).__init__()

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ['channels_last', 'channels_first']:
            raise NotImplementedError
        self.normalized_shape = (normalized_shape,)

    def forward(self, x):

        if self.data_format == 'channels_last':
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == 'channels_first':
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x


class Block(Net):

    def __init__(self, dim, drop_path=0., layer_scale_init_value=1e-6):

        super(Block, self).__init__()

        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)
        self.norm = LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, 4 * dim)
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones((dim)), requires_grad=True) if layer_scale_init_value > 0 else None
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.weights_init()

    def forward(self, x):

        inputs = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma * x
        x = x.permute(0, 3, 1, 2)
        x = inputs + self.drop_path(x)
        return x


class UpSample2x(nn.Module):

    def __init__(self):

        super(UpSample2x, self).__init__()
        self.register_buffer('unpool_mat', torch.ones((2, 2), dtype=torch.float32))

    def forward(self, x):
        input_shape = x.shape
        x = x.unsqueeze(-1)
        mat = self.unpool_mat.unsqueeze(0)
        ret = torch.tensordot(x, mat, dims=1)
        ret = ret.permute(0, 1, 2, 4, 3, 5)
        ret = ret.reshape((-1, input_shape[1], input_shape[2] * 2, input_shape[3] * 2))
        return ret


class ConvNeXt(Net):

    def __init__(self, in_chans=3, depths=(3, 3, 9, 3), dims=(96, 192, 384, 768), drop_path_rate=0., layer_scale_init_value=1e-6):

        super(ConvNeXt, self).__init__()

        self.downsample_layers = nn.ModuleList()
        stem = nn.Sequential(
            nn.Conv2d(in_chans, dims[0], kernel_size=3, stride=1, padding='same'),
            LayerNorm(dims[0], eps=1e-6, data_format='channels_first')
        )
        self.downsample_layers.append(stem)
        for i in range(3):
            downsample_layer = nn.Sequential(
                LayerNorm(dims[i], eps=1e-6, data_format='channels_first'),
                nn.Conv2d(dims[i], dims[i + 1], kernel_size=2, stride=2),
            )
            self.downsample_layers.append(downsample_layer)

        self.stages = nn.ModuleList()
        dp_rates = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        cur = 0
        for i in range(4):
            stage = nn.Sequential(
                *[Block(
                    dim=dims[i],
                    drop_path=dp_rates[cur + j],
                    layer_scale_init_value=layer_scale_init_value
                ) for j in range(depths[i])],
                LayerNorm(dims[i], eps=1e-6, data_format='channels_first')
            )

            self.stages.append(stage)
            cur += depths[i]
        self.last_layer = nn.Conv2d(dims[-1], dims[-2], kernel_size=3, stride=1, padding='same')
        self.weights_init()

    def forward_features(self, x):
        feat_maps = []
        for i in range(4):
            x = self.downsample_layers[i](x)
            x = self.stages[i](x)
            if i == 3:
                x = self.last_layer(x)
            feat_maps.append(x)
        return feat_maps

    def forward(self, x):

        x = self.forward_features(x)
        return x


class TFSamepaddingLayer(nn.Module):

    def __init__(self, ksize, stride):

        super(TFSamepaddingLayer, self).__init__()

        self.ksize = ksize
        self.stride = stride

    def forward(self, x):

        if x.shape[2] % self.stride == 0:
            pad = max(self.ksize - self.stride, 0)
        else:
            pad = max(self.ksize - (x.shape[2] % self.stride), 0)

        if pad % 2 == 0:
            pad_val = pad // 2
            padding = (pad_val, pad_val, pad_val, pad_val)
        else:
            pad_val_start = pad // 2
            pad_val_end = pad - pad_val_start
            padding = (pad_val_start, pad_val_end, pad_val_start, pad_val_end)

        x = F.pad(x, padding, 'constant', 0)
        return x


class DenseBlock(Net):

    def __init__(self, in_ch, unit_ksize, unit_ch, unit_count, split=1):

        super(DenseBlock, self).__init__()
        assert len(unit_ksize) == len(unit_ch), 'Unbalance Unit Info'

        self.nr_unit = unit_count
        self.in_ch = in_ch
        self.unit_ch = unit_ch

        unit_in_ch = in_ch
        self.units = nn.ModuleList()
        for idx in range(unit_count):
            self.units.append(
                nn.Sequential(
                    OrderedDict(
                        [
                            ('preact_bna/bn', nn.BatchNorm2d(unit_in_ch, eps=1e-5)),
                            ('preact_bna/relu', nn.ReLU(inplace=True)),
                            (
                                'conv1',
                                nn.Conv2d(
                                    unit_in_ch,
                                    unit_ch[0],
                                    unit_ksize[0],
                                    stride=1,
                                    padding=0,
                                    bias=False,
                                ),
                            ),
                            ('conv1/bn', nn.BatchNorm2d(unit_ch[0], eps=1e-5)),
                            ('conv1/relu', nn.ReLU(inplace=True)),
                            (
                                'conv2',
                                nn.Conv2d(
                                    unit_ch[0],
                                    unit_ch[1],
                                    unit_ksize[1],
                                    groups=split,
                                    stride=1,
                                    padding=0,
                                    bias=False,
                                ),
                            ),
                        ]
                    )
                )
            )
            unit_in_ch += unit_ch[1]

        self.blk_bna = nn.Sequential(
            OrderedDict(
                [
                    ('bn', nn.BatchNorm2d(unit_in_ch, eps=1e-5)),
                    ('relu', nn.ReLU(inplace=True)),
                ]
            )
        )

    def out_ch(self):
        return self.in_ch + self.nr_unit * self.unit_ch[-1]

    def forward(self, prev_feat):
        for idx in range(self.nr_unit):
            new_feat = self.units[idx](prev_feat)
            prev_feat = crop_to_shape(prev_feat, new_feat)
            prev_feat = torch.cat([prev_feat, new_feat], dim=1)
        prev_feat = self.blk_bna(prev_feat)

        return prev_feat


class HoVerNet(Net):

    def __init__(self, out_ch=1):

        super(HoVerNet, self).__init__()

        self.encoder = ConvNeXt(depths=[1, 1, 1, 1])

        def create_decoder_branch(out_ch=out_ch, ksize=5):

            module_list = [
                ('conva', nn.Conv2d(384, 256, ksize, stride=1, padding=0, bias=False)),
                ('dense', DenseBlock(256, [1, ksize], [128, 32], 8, split=4)),
                ('convf', nn.Conv2d(512, 192, 1, stride=1, padding=0, bias=False),),
            ]
            u3 = nn.Sequential(OrderedDict(module_list))

            module_list = [
                ('conva', nn.Conv2d(192, 128, ksize, stride=1, padding=0, bias=False)),
                ('dense', DenseBlock(128, [1, ksize], [128, 32], 4, split=4)),
                ('convf', nn.Conv2d(256, 96, 1, stride=1, padding=0, bias=False),),
            ]
            u2 = nn.Sequential(OrderedDict(module_list))

            module_list = [
                ('conva/pad', TFSamepaddingLayer(ksize=ksize, stride=1)),
                ('conva', nn.Conv2d(96, 64, ksize, stride=1, padding=0, bias=False),),
            ]
            u1 = nn.Sequential(OrderedDict(module_list))

            module_list = [
                ('bn', nn.BatchNorm2d(64, eps=1e-5)),
                ('relu', nn.ReLU(inplace=True)),
                ('conv', nn.Conv2d(64, out_ch, 1, stride=1, padding=0, bias=True),),
            ]
            u0 = nn.Sequential(OrderedDict(module_list))

            decoder = nn.Sequential(
                OrderedDict([('u3', u3), ('u2', u2), ('u1', u1), ('u0', u0)])
            )
            return decoder

        self.decoder = nn.ModuleDict(
            OrderedDict(
                [
                    ('np', create_decoder_branch(ksize=3, out_ch=2)),
                    ('hv', create_decoder_branch(ksize=3, out_ch=2)),
                ]
            )
        )

        self.upsample2x = UpSample2x()

    def forward(self, x):

        d = self.encoder(x)

        d[0] = crop_op(d[0], [92, 92])
        d[1] = crop_op(d[1], [36, 36])

        out_dict = OrderedDict()
        for branch_name, branch_desc in self.decoder.items():
            u3 = self.upsample2x(d[-1]) + d[-2]
            u3 = branch_desc[0](u3)

            u2 = self.upsample2x(u3) + d[-3]
            u2 = branch_desc[1](u2)

            u1 = self.upsample2x(u2) + d[-4]
            u1 = branch_desc[2](u1)

            u0 = branch_desc[3](u1)
            out_dict[branch_name] = u0

        return out_dict
