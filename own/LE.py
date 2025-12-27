import torch
import torch.nn as nn

import pywt
import pywt.data
import torch
from torch import nn
from functools import partial
import torch.nn.functional as F
# 论文地址 https://arxiv.org/pdf/2407.05848
def create_wavelet_filter(wave, in_size, out_size, type=torch.float):
    w = pywt.Wavelet(wave)
    dec_hi = torch.tensor(w.dec_hi[::-1], dtype=type)
    dec_lo = torch.tensor(w.dec_lo[::-1], dtype=type)
    dec_filters = torch.stack([dec_lo.unsqueeze(0) * dec_lo.unsqueeze(1),
                               dec_lo.unsqueeze(0) * dec_hi.unsqueeze(1),
                               dec_hi.unsqueeze(0) * dec_lo.unsqueeze(1),
                               dec_hi.unsqueeze(0) * dec_hi.unsqueeze(1)], dim=0)

    dec_filters = dec_filters[:, None].repeat(in_size, 1, 1, 1)

    rec_hi = torch.tensor(w.rec_hi[::-1], dtype=type).flip(dims=[0])
    rec_lo = torch.tensor(w.rec_lo[::-1], dtype=type).flip(dims=[0])
    rec_filters = torch.stack([rec_lo.unsqueeze(0) * rec_lo.unsqueeze(1),
                               rec_lo.unsqueeze(0) * rec_hi.unsqueeze(1),
                               rec_hi.unsqueeze(0) * rec_lo.unsqueeze(1),
                               rec_hi.unsqueeze(0) * rec_hi.unsqueeze(1)], dim=0)

    rec_filters = rec_filters[:, None].repeat(out_size, 1, 1, 1)

    return dec_filters, rec_filters


def wavelet_transform(x, filters):
    b, c, h, w = x.shape
    pad = (filters.shape[2] // 2 - 1, filters.shape[3] // 2 - 1)
    x = F.conv2d(x, filters, stride=2, groups=c, padding=pad)
    x = x.reshape(b, c, 4, h // 2, w // 2)
    return x


def inverse_wavelet_transform(x, filters):
    b, c, _, h_half, w_half = x.shape
    pad = (filters.shape[2] // 2 - 1, filters.shape[3] // 2 - 1)
    x = x.reshape(b, c * 4, h_half, w_half)
    x = F.conv_transpose2d(x, filters, stride=2, groups=c, padding=pad)
    return x


# Wavelet Transform Conv(WTConv2d)
class WTConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=5, stride=1, bias=True, wt_levels=1, wt_type='db1'):
        super(WTConv2d, self).__init__()

        assert in_channels == out_channels

        self.in_channels = in_channels
        self.wt_levels = wt_levels
        self.stride = stride
        self.dilation = 1

        self.wt_filter, self.iwt_filter = create_wavelet_filter(wt_type, in_channels, in_channels, torch.float)
        self.wt_filter = nn.Parameter(self.wt_filter, requires_grad=False)
        self.iwt_filter = nn.Parameter(self.iwt_filter, requires_grad=False)

        self.wt_function = partial(wavelet_transform, filters=self.wt_filter)
        self.iwt_function = partial(inverse_wavelet_transform, filters=self.iwt_filter)

        self.base_conv = nn.Conv2d(in_channels, in_channels, kernel_size, padding='same', stride=1, dilation=1,
                                   groups=in_channels, bias=bias)
        self.base_scale = _ScaleModule([1, in_channels, 1, 1])

        self.wavelet_convs = nn.ModuleList(
            [nn.Conv2d(in_channels * 4, in_channels * 4, kernel_size, padding='same', stride=1, dilation=1,
                       groups=in_channels * 4, bias=False) for _ in range(self.wt_levels)]
        )
        self.wavelet_scale = nn.ModuleList(
            [_ScaleModule([1, in_channels * 4, 1, 1], init_scale=0.1) for _ in range(self.wt_levels)]
        )

        if self.stride > 1:
            self.stride_filter = nn.Parameter(torch.ones(in_channels, 1, 1, 1), requires_grad=False)
            self.do_stride = lambda x_in: F.conv2d(x_in, self.stride_filter, bias=None, stride=self.stride,
                                                   groups=in_channels)
        else:
            self.do_stride = None

    def forward(self, x):

        x_ll_in_levels = []
        x_h_in_levels = []
        shapes_in_levels = []

        curr_x_ll = x

        for i in range(self.wt_levels):
            curr_shape = curr_x_ll.shape
            shapes_in_levels.append(curr_shape)
            if (curr_shape[2] % 2 > 0) or (curr_shape[3] % 2 > 0):
                curr_pads = (0, curr_shape[3] % 2, 0, curr_shape[2] % 2)
                curr_x_ll = F.pad(curr_x_ll, curr_pads)

            curr_x = self.wt_function(curr_x_ll)
            curr_x_ll = curr_x[:, :, 0, :, :]

            shape_x = curr_x.shape
            curr_x_tag = curr_x.reshape(shape_x[0], shape_x[1] * 4, shape_x[3], shape_x[4])
            curr_x_tag = self.wavelet_scale[i](self.wavelet_convs[i](curr_x_tag))
            curr_x_tag = curr_x_tag.reshape(shape_x)

            x_ll_in_levels.append(curr_x_tag[:, :, 0, :, :])
            x_h_in_levels.append(curr_x_tag[:, :, 1:4, :, :])

        next_x_ll = 0

        for i in range(self.wt_levels - 1, -1, -1):
            curr_x_ll = x_ll_in_levels.pop()
            curr_x_h = x_h_in_levels.pop()
            curr_shape = shapes_in_levels.pop()

            curr_x_ll = curr_x_ll + next_x_ll

            curr_x = torch.cat([curr_x_ll.unsqueeze(2), curr_x_h], dim=2)
            next_x_ll = self.iwt_function(curr_x)

            next_x_ll = next_x_ll[:, :, :curr_shape[2], :curr_shape[3]]

        x_tag = next_x_ll
        assert len(x_ll_in_levels) == 0

        x = self.base_scale(self.base_conv(x))
        x = x + x_tag

        if self.do_stride is not None:
            x = self.do_stride(x)

        return x

class _ScaleModule(nn.Module):
    def __init__(self, dims, init_scale=1.0, init_bias=0):
        super(_ScaleModule, self).__init__()
        self.dims = dims
        self.weight = nn.Parameter(torch.ones(*dims) * init_scale)
        self.bias = None

    def forward(self, x):
        return torch.mul(self.weight, x)


class DepthwiseSeparableConvWithWTConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3):
        super(DepthwiseSeparableConvWithWTConv2d, self).__init__()

        # 深度卷积：使用 WTConv2d 替换 3x3 卷积
        self.depthwise = WTConv2d(in_channels, in_channels, kernel_size=kernel_size)

        # 逐点卷积：使用 1x1 卷积
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x
#################################################################################
def swish(x):
    return x * x.sigmoid()

def hard_sigmoid(x, inplace=False):
    return nn.ReLU6(inplace=inplace)(x + 3) / 6

def hard_swish(x, inplace=False):
    return x * hard_sigmoid(x, inplace)

class HardSigmoid(nn.Module):
    def __init__(self, inplace=False):
        super(HardSigmoid, self).__init__()
        self.inplace = inplace

    def forward(self, x):
        return hard_sigmoid(x, inplace=self.inplace)

class HardSwish(nn.Module):
    def __init__(self, inplace=False):
        super(HardSwish, self).__init__()
        self.inplace = inplace

    def forward(self, x):
        return hard_swish(x, inplace=self.inplace)

def _make_divisible(v, divisor=8, min_value=None):  ## 将通道数变成8的整数倍
    """
    This function is taken from the original tf repo.
    It ensures that all layers have a channel number that is divisible by 8
    It can be seen here:
    https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    :param v:
    :param divisor:
    :param min_value:
    :return:
    """
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


class SELayer(nn.Module):
    def __init__(self, inp, oup, reduction=4):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
                nn.Conv2d(oup, _make_divisible(inp // reduction), 1, 1, 0,),
                nn.ReLU(),
                nn.Conv2d(_make_divisible(inp // reduction), oup, 1, 1, 0),
                HardSigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y

class ConvBlock1(nn.Module):
    def __init__(self):
        super(ConvBlock1, self).__init__()
        self.DW = nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, stride=1, groups=16, padding=1, bias=False)
        self.BN = nn.BatchNorm2d(16)
        self.HS = HardSwish()
        self.PW = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=1, stride=1, padding=0, bias=False)
        self.BNN = nn.BatchNorm2d(32)

    def forward(self, x):
        a = self.HS(self.BN(self.DW(x)))
        a = self.HS(self.BNN(self.PW(a)))
        return a

class ConvBlock2(nn.Module):
    def __init__(self):
        super(ConvBlock2, self).__init__()
        self.DW = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, groups=32, padding=1, bias=False)
        self.BN = nn.BatchNorm2d(32)
        self.HS = HardSwish()
        self.PW = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=1, stride=1, padding=0, bias=False)
        self.BNN = nn.BatchNorm2d(64)

    def forward(self, x):
        a = self.HS(self.BN(self.DW(x)))
        a = self.HS(self.BNN(self.PW(a)))
        return a

class ConvBlock3(nn.Module):
    def __init__(self):
        super(ConvBlock3, self).__init__()
        self.DW = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, groups=64, padding=1, bias=False)
        self.BN = nn.BatchNorm2d(64)
        self.HS = HardSwish()
        self.PW = nn.Conv2d(in_channels=64, out_channels=32, kernel_size=1, stride=1, padding=0, bias=False)
        self.BNN = nn.BatchNorm2d(32)

    def forward(self, x):
        a = self.HS(self.BN(self.DW(x)))
        a = self.HS(self.BNN(self.PW(a)))
        return a

class ConvBlock4(nn.Module):
    def __init__(self):
        super(ConvBlock4, self).__init__()
        self.DW = nn.Conv2d(in_channels=80, out_channels=80, kernel_size=3, stride=1, groups=80, padding=1, bias=False)
        self.BN = nn.BatchNorm2d(80)
        self.HS = HardSwish()
        self.PW = nn.Conv2d(in_channels=80, out_channels=32, kernel_size=1, stride=1, padding=0, bias=False)
        self.BNN = nn.BatchNorm2d(32)
        self.SE = SELayer(80, 80)

    def forward(self, x):

        a = self.HS(self.BN(self.DW(x)))
        a = self.SE(a)
        a = self.HS(self.BNN(self.PW(a)))
        return a

class Mynet(nn.Module):
    def __init__(self, num_layers=3):
        super(Mynet, self).__init__()
        self.input = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=1, stride=1, padding=0, bias=False)  ## 第一层卷积
        self.output = nn.Conv2d(in_channels=32, out_channels=3, kernel_size=1, stride=1, padding=0, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.block1 = ConvBlock1()
        self.block2 = ConvBlock2()
        self.block3 = ConvBlock3()
        self.block4 = ConvBlock4()

    def forward(self, x):
        x = self.input(x)
        x1 = self.block1(x)
        x2 = self.block2(x1)
        # x2 = torch.cat((x, x2), 1)
        x3 = self.block3(x2)
        x3 = torch.cat((x, x1, x3), 1)
        x4 = self.block4(x3)
        out = self.output(x4)
        return out

class MynetWTC(nn.Module):
    def __init__(self, num_layers=3):
        super(MynetWTC, self).__init__()
        self.input = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=1, stride=1, padding=0, bias=False)  ## 第一层卷积
        self.output = nn.Conv2d(in_channels=32, out_channels=3, kernel_size=1, stride=1, padding=0, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.block1 = ConvBlock1()
        self.block2 = ConvBlock2()
        self.block3 = ConvBlock3()
        self.block4 = ConvBlock4()
        self.wtc = DepthwiseSeparableConvWithWTConv2d(32,32)

    def forward(self, x):
        x = self.input(x)
        x1 = self.block1(x)
        ####################
        x5 = self.wtc(x1)
        ###########################
        x2 = self.block2(x1)
        # x2 = torch.cat((x, x2), 1)
        x3 = self.block3(x2)
        x3 = torch.cat((x, x1, x3), 1)
        x4 = self.block4(x3)
        out = self.output(x4+x5)
        return out

