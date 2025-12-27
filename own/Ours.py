import torch
import torch.nn as nn

"""
https://arxiv.org/abs/2303.03667
<<Run, Don't Walk: Chasing Higher FLOPS for Faster Neural Networks>>
"""

#######################################################################LNet TB###################################################################

class Partial_conv3(nn.Module):
    def __init__(self, dim, n_div, forward):
        super().__init__()
        self.dim_conv3 = dim // n_div
        self.dim_untouched = dim - self.dim_conv3
        self.partial_conv3 = nn.Conv2d(self.dim_conv3, self.dim_conv3, 3, 1, 1, bias=False)

        if forward == 'slicing':
            self.forward = self.forward_slicing
        elif forward == 'split_cat':
            self.forward = self.forward_split_cat
        else:
            raise NotImplementedError

    def forward_slicing(self, x):
        # only for inference
        x = x.clone()  # !!! Keep the original input intact for the residual connection later
        x[:, :self.dim_conv3, :, :] = self.partial_conv3(x[:, :self.dim_conv3, :, :])

        return x

    def forward_split_cat(self, x):
        # for training/inference
        x1, x2 = torch.split(x, [self.dim_conv3, self.dim_untouched], dim=1)
        x1 = self.partial_conv3(x1)
        x = torch.cat((x1, x2), 1)
        return x






class LNet(torch.nn.Module):
    def __init__(self, num=64):
        super().__init__()
        self.conv1 = torch.nn.Sequential(
            torch.nn.ReflectionPad2d(1),
            torch.nn.Conv2d(3, num, 3, 1, 0),
            torch.nn.InstanceNorm2d(num),
            torch.nn.ReLU()
        )
        self.conv2 = torch.nn.Sequential(
            # torch.nn.ReflectionPad2d(1),
            Partial_conv3(num,num,forward='split_cat'),
            torch.nn.InstanceNorm2d(num),
            torch.nn.ReLU()
        )
        self.conv3 = torch.nn.Sequential(
            # torch.nn.ReflectionPad2d(1),
            Partial_conv3(num,num,forward='split_cat'),
            # torch.nn.Conv2d(num, num, 3, 1, 0),
            torch.nn.InstanceNorm2d(num),
            torch.nn.ReLU()
        )
        self.conv4 = torch.nn.Sequential(
            # torch.nn.ReflectionPad2d(1),
            Partial_conv3(num,num,forward='split_cat'),
            torch.nn.InstanceNorm2d(num),
            torch.nn.ReLU()
        )
        self.final = torch.nn.Sequential(
            torch.nn.Conv2d(num, 3, 1, 1, 0),
            torch.nn.Sigmoid()
        )

    def forward(self, data):
        data = self.conv1(data)
        data = self.conv2(data)
        data = self.conv3(data)
        data = self.conv4(data)
        data1 = self.final(data)

        return data1

################################################################################################################################################

#############################################################LWTNet    T#####################################################################
import pywt
import pywt.data
import torch
from torch import nn
from functools import partial
import torch.nn.functional as F
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


class LWTNet(torch.nn.Module):
    def __init__(self, num=64):
        super().__init__()
        self.conv1 = torch.nn.Sequential(
            torch.nn.ReflectionPad2d(1),
            torch.nn.Conv2d(3, num, 3, 1, 0),
            torch.nn.InstanceNorm2d(num),
            torch.nn.ReLU()
        )
        self.conv2 = torch.nn.Sequential(
            # torch.nn.ReflectionPad2d(1),
            DepthwiseSeparableConvWithWTConv2d(num,num),
            torch.nn.InstanceNorm2d(num),
            torch.nn.ReLU()
        )
        self.conv3 = torch.nn.Sequential(
            # torch.nn.ReflectionPad2d(1),
            DepthwiseSeparableConvWithWTConv2d(num,num),
            # torch.nn.Conv2d(num, num, 3, 1, 0),
            torch.nn.InstanceNorm2d(num),
            torch.nn.ReLU()
        )
        self.conv4 = torch.nn.Sequential(
            # torch.nn.ReflectionPad2d(1),
            DepthwiseSeparableConvWithWTConv2d(num,num),
            torch.nn.InstanceNorm2d(num),
            torch.nn.ReLU()
        )
        self.final = torch.nn.Sequential(
            torch.nn.Conv2d(num, 3, 1, 1, 0),
            torch.nn.Sigmoid()
        )

    def forward(self, data):
        data = self.conv1(data)
        data = self.conv2(data)
        data = self.conv3(data)
        data = self.conv4(data)
        data1 = self.final(data)

        return data1

##################################################################################################################################
#############################################################zhuyili#####################################################################
class FourierUnit(nn.Module):
    def __init__(self, in_channels, out_channels, groups=1):
        super(FourierUnit, self).__init__()
        self.groups = groups
        self.conv_layer = torch.nn.Conv2d(in_channels=in_channels * 2, out_channels=out_channels * 2,
                                          kernel_size=1, stride=1, padding=0, groups=self.groups, bias=False)
        self.bn = torch.nn.BatchNorm2d(out_channels * 2)
        self.relu = torch.nn.ReLU(inplace=True)

    def forward(self, x):
        batch, c, h, w = x.size()
        # (batch, c, h, w/2+1) 复数
        ffted = torch.fft.rfftn(x, s=(h, w), dim=(2, 3), norm='ortho')
        ffted = torch.cat([ffted.real, ffted.imag], dim=1)

        ffted = self.conv_layer(ffted)  # (batch, c*2, h, w/2+1)
        ffted = self.relu(self.bn(ffted))

        ffted = torch.tensor_split(ffted, 2, dim=1)
        ffted = torch.complex(ffted[0], ffted[1])
        output = torch.fft.irfftn(ffted, s=(h, w), dim=(2, 3), norm='ortho')
        return output


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

        self.fu = FourierUnit(2, 1)

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)

        fu_out = self.fu(x)

        x = self.conv1(x + fu_out)
        return self.sigmoid(x)


class Flatten(nn.Module):
    def forward(self, inp):
        return inp.view(inp.size(0), -1)


class CAM(nn.Module):
    def __init__(self, in_channels, reduction_ratio):
        super(CAM, self).__init__()
        self.module = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            Flatten(),
            nn.Linear(in_channels, in_channels // reduction_ratio),
            nn.Softsign(),
            nn.Linear(in_channels // reduction_ratio, in_channels),
            nn.Softsign()
        )
        self.sa = SpatialAttention(7)

    def forward(self, input):
        return input * self.module(input).unsqueeze(2).unsqueeze(3).expand_as(input)+self.sa(input)





class fu(nn.Module):
    def __init__(self,dim):
        super().__init__()
        self.conv1 = nn.Conv2d(dim,dim,1)
        self.conv2 = nn.Conv2d(dim, dim, 1)
        self.ffc = CAM(dim,2)   #############w/ofcsam
        self.act = nn.ReLU()

    def forward(self,x):
        res = x
        x = self.conv1(x)
        x = self.ffc(x)   #############w/ofcsam
        x = self.act(x)
        x = self.conv2(x)
        return  x + res



################################################################################################################################################

############################################################Conv-Mamba  Jnet########################################################################################################
def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution."""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


####################################################SS2D#####################################################################
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import repeat

try:
    from mamba_ssm.ops.selective_scan_interface import selective_scan_fn
except:
    pass

# an alternative for mamba_ssm (in which causal_conv1d is needed)
try:
    from selective_scan import selective_scan_fn as selective_scan_fn_v1
except:
    pass


class SS2D(nn.Module):
    def __init__(
            self,
            d_model,
            d_state=16,
            # d_state="auto", # 20240109
            d_conv=3,
            expand=2,
            dt_rank="auto",
            dt_min=0.001,
            dt_max=0.1,
            dt_init="random",
            dt_scale=1.0,
            dt_init_floor=1e-4,
            dropout=0.,
            conv_bias=True,
            bias=False,
            device=None,
            dtype=None,
            **kwargs,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        # self.d_state = math.ceil(self.d_model / 6) if d_state == "auto" else d_model # 20240109
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank

        self.in_proj = nn.Linear(self.d_model, self.d_inner * 2, bias=bias, **factory_kwargs)
        self.conv2d = nn.Conv2d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            groups=self.d_inner,
            bias=conv_bias,
            kernel_size=d_conv,
            padding=(d_conv - 1) // 2,
            **factory_kwargs,
        )
        self.act = nn.SiLU()

        self.x_proj = (
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
        )
        self.x_proj_weight = nn.Parameter(torch.stack([t.weight for t in self.x_proj], dim=0))  # (K=4, N, inner)
        del self.x_proj

        self.dt_projs = (
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor,
                         **factory_kwargs),
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor,
                         **factory_kwargs),
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor,
                         **factory_kwargs),
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor,
                         **factory_kwargs),
        )
        self.dt_projs_weight = nn.Parameter(torch.stack([t.weight for t in self.dt_projs], dim=0))  # (K=4, inner, rank)
        self.dt_projs_bias = nn.Parameter(torch.stack([t.bias for t in self.dt_projs], dim=0))  # (K=4, inner)
        del self.dt_projs

        self.A_logs = self.A_log_init(self.d_state, self.d_inner, copies=4, merge=True)  # (K=4, D, N)
        self.Ds = self.D_init(self.d_inner, copies=4, merge=True)  # (K=4, D, N)

        # self.selective_scan = selective_scan_fn
        self.forward_core = self.forward_corev0

        self.out_norm = nn.LayerNorm(self.d_inner)
        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias, **factory_kwargs)
        self.dropout = nn.Dropout(dropout) if dropout > 0. else None

    @staticmethod
    def dt_init(dt_rank, d_inner, dt_scale=1.0, dt_init="random", dt_min=0.001, dt_max=0.1, dt_init_floor=1e-4,
                **factory_kwargs):
        dt_proj = nn.Linear(dt_rank, d_inner, bias=True, **factory_kwargs)

        # Initialize special dt projection to preserve variance at initialization
        dt_init_std = dt_rank ** -0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError

        # Initialize dt bias so that F.softplus(dt_bias) is between dt_min and dt_max
        dt = torch.exp(
            torch.rand(d_inner, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        # Inverse of softplus: https://github.com/pytorch/pytorch/issues/72759
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            dt_proj.bias.copy_(inv_dt)
        # Our initialization would set all Linear.bias to zero, need to mark this one as _no_reinit
        dt_proj.bias._no_reinit = True

        return dt_proj

    @staticmethod
    def A_log_init(d_state, d_inner, copies=1, device=None, merge=True):
        # S4D real initialization
        A = repeat(
            torch.arange(1, d_state + 1, dtype=torch.float32, device=device),
            "n -> d n",
            d=d_inner,
        ).contiguous()
        A_log = torch.log(A)  # Keep A_log in fp32
        if copies > 1:
            A_log = repeat(A_log, "d n -> r d n", r=copies)
            if merge:
                A_log = A_log.flatten(0, 1)
        A_log = nn.Parameter(A_log)
        A_log._no_weight_decay = True
        return A_log

    @staticmethod
    def D_init(d_inner, copies=1, device=None, merge=True):
        # D "skip" parameter
        D = torch.ones(d_inner, device=device)
        if copies > 1:
            D = repeat(D, "n1 -> r n1", r=copies)
            if merge:
                D = D.flatten(0, 1)
        D = nn.Parameter(D)  # Keep in fp32
        D._no_weight_decay = True
        return D

    def forward_corev0(self, x: torch.Tensor):
        self.selective_scan = selective_scan_fn

        # x tranform form (b, d, h, w)
        B, C, H, W = x.shape
        L = H * W
        K = 4

        x_hwwh = torch.stack([x.view(B, -1, L), torch.transpose(x, dim0=2, dim1=3).contiguous().view(B, -1, L)],
                             dim=1).view(B, 2, -1, L)
        xs = torch.cat([x_hwwh, torch.flip(x_hwwh, dims=[-1])], dim=1)  # (b, k, d, l)

        x_dbl = torch.einsum("b k d l, k c d -> b k c l", xs.view(B, K, -1, L), self.x_proj_weight)
        # x_dbl = x_dbl + self.x_proj_bias.view(1, K, -1, 1)
        dts, Bs, Cs = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=2)
        dts = torch.einsum("b k r l, k d r -> b k d l", dts.view(B, K, -1, L), self.dt_projs_weight)
        # dts = dts + self.dt_projs_bias.view(1, K, -1, 1)

        xs = xs.float().view(B, -1, L)  # (b, k * d, l)
        dts = dts.contiguous().float().view(B, -1, L)  # (b, k * d, l)
        Bs = Bs.float().view(B, K, -1, L)  # (b, k, d_state, l)
        Cs = Cs.float().view(B, K, -1, L)  # (b, k, d_state, l)
        Ds = self.Ds.float().view(-1)  # (k * d)
        As = -torch.exp(self.A_logs.float()).view(-1, self.d_state)  # (k * d, d_state)
        dt_projs_bias = self.dt_projs_bias.float().view(-1)  # (k * d)

        out_y = self.selective_scan(
            xs, dts,
            As, Bs, Cs, Ds, z=None,
            delta_bias=dt_projs_bias,
            delta_softplus=True,
            return_last_state=False,
        ).view(B, K, -1, L)
        assert out_y.dtype == torch.float

        inv_y = torch.flip(out_y[:, 2:4], dims=[-1]).view(B, 2, -1, L)
        wh_y = torch.transpose(out_y[:, 1].view(B, -1, W, H), dim0=2, dim1=3).contiguous().view(B, -1, L)
        invwh_y = torch.transpose(inv_y[:, 1].view(B, -1, W, H), dim0=2, dim1=3).contiguous().view(B, -1, L)

        return out_y[:, 0], inv_y[:, 0], wh_y, invwh_y

    # an alternative to forward_corev1
    def forward_corev1(self, x: torch.Tensor):
        self.selective_scan = selective_scan_fn_v1

        B, C, H, W = x.shape
        L = H * W
        K = 4

        x_hwwh = torch.stack([x.view(B, -1, L), torch.transpose(x, dim0=2, dim1=3).contiguous().view(B, -1, L)],
                             dim=1).view(B, 2, -1, L)
        xs = torch.cat([x_hwwh, torch.flip(x_hwwh, dims=[-1])], dim=1)  # (b, k, d, l)

        x_dbl = torch.einsum("b k d l, k c d -> b k c l", xs.view(B, K, -1, L), self.x_proj_weight)
        # x_dbl = x_dbl + self.x_proj_bias.view(1, K, -1, 1)
        dts, Bs, Cs = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=2)
        dts = torch.einsum("b k r l, k d r -> b k d l", dts.view(B, K, -1, L), self.dt_projs_weight)
        # dts = dts + self.dt_projs_bias.view(1, K, -1, 1)

        xs = xs.float().view(B, -1, L)  # (b, k * d, l)
        dts = dts.contiguous().float().view(B, -1, L)  # (b, k * d, l)
        Bs = Bs.float().view(B, K, -1, L)  # (b, k, d_state, l)
        Cs = Cs.float().view(B, K, -1, L)  # (b, k, d_state, l)
        Ds = self.Ds.float().view(-1)  # (k * d)
        As = -torch.exp(self.A_logs.float()).view(-1, self.d_state)  # (k * d, d_state)
        dt_projs_bias = self.dt_projs_bias.float().view(-1)  # (k * d)

        out_y = self.selective_scan(
            xs, dts,
            As, Bs, Cs, Ds,
            delta_bias=dt_projs_bias,
            delta_softplus=True,
        ).view(B, K, -1, L)
        assert out_y.dtype == torch.float

        inv_y = torch.flip(out_y[:, 2:4], dims=[-1]).view(B, 2, -1, L)
        wh_y = torch.transpose(out_y[:, 1].view(B, -1, W, H), dim0=2, dim1=3).contiguous().view(B, -1, L)
        invwh_y = torch.transpose(inv_y[:, 1].view(B, -1, W, H), dim0=2, dim1=3).contiguous().view(B, -1, L)

        return out_y[:, 0], inv_y[:, 0], wh_y, invwh_y

    def forward(self, x: torch.Tensor, **kwargs):
        x = x.permute(0, 2, 3, 1).contiguous()
        B, H, W, C = x.shape

        xz = self.in_proj(x)
        x, z = xz.chunk(2, dim=-1)  # (b, h, w, d)

        x = x.permute(0, 3, 1, 2).contiguous()
        x = self.act(self.conv2d(x))  # (b, d, h, w)
        y1, y2, y3, y4 = self.forward_core(x)
        assert y1.dtype == torch.float32
        y = y1 + y2 + y3 + y4
        y = torch.transpose(y, dim0=1, dim1=2).contiguous().view(B, H, W, -1)
        try:
            y = self.out_norm(y)
        except:
            y = self.out_norm.to(torch.float32)(y).half()

        y = y * F.silu(z)
        try:
            out = self.out_proj(y)
        except:
            out = self.out_proj.to(torch.float32)(y).half()
        if self.dropout is not None:
            out = self.dropout(out)
        out = out.permute(0, 3, 1, 2).contiguous()
        return out
#######################################################################################################################################################
class SSD1(nn.Module):
    def __init__(self,dim1,dim2):
        super().__init__()
        self.conv1 = nn.Conv2d(dim1,dim2,3,1,1)
        self.ssd = SS2D(dim2)
        self.IN  = nn.InstanceNorm2d(dim2)
        self.act = nn.ReLU()
        self.mlp = nn.Conv2d(dim2,dim2,1)

    def forward(self,x):
        res = x
        x = self.conv1(x)
        x = self.act(x)
        x = self.ssd(x)
        res2 = res + x
        x = self.IN(res2)
        x = self.mlp(x)
        x = res2 + x
        return x+res


class SSD2(nn.Module):
    def __init__(self,dim1,dim2):
        super().__init__()
        self.conv1 =  torch.nn.Sequential(
            torch.nn.ReflectionPad2d(1),
            torch.nn.Conv2d(dim1, dim2, 3, 1, 0),
            torch.nn.InstanceNorm2d(dim2),
            torch.nn.ReLU()
        )
        self.conv2 = conv1x1(dim2,dim2)
        self.ssd = SS2D(dim2)
        self.IN  = nn.InstanceNorm2d(dim2)
        self.act = nn.ReLU()
        self.mlp = nn.Conv2d(dim2,dim2,1)

    def forward(self,x):
        res = self.conv1(x)
        x = self.conv2(res)
        x = self.act(x)
        x = self.ssd(x)
        res2 = res + x
        x = self.IN(res2)
        x = self.mlp(x)
        return x

class SSD3(nn.Module):
    def __init__(self,dim1,dim2):
        super().__init__()
        self.conv1 = nn.Conv2d(dim1,dim2,3,1,1)
        self.ssd = SS2D(dim2)
        self.IN  = nn.InstanceNorm2d(dim2)
        self.act = nn.ReLU()

    def forward(self,x):
        res = x
        x = self.IN(x)
        x = self.conv1(x)
        x = self.act(x)
        x = self.ssd(x)
        x = self.IN(x)
        return x + res

class SSD4(nn.Module):
    def __init__(self,dim1,dim2):
        super().__init__()
        self.conv11 = nn.Conv2d(dim1,dim2,1)
        self.conv33 = nn.Conv2d(dim2,dim2,3,1,1)
        self.ssd = SS2D(dim2)
        self.IN  = nn.InstanceNorm2d(dim1)
        self.act = nn.ReLU()
        self.conv111 = nn.Conv2d(dim2,dim2,1)

    def forward(self,x):
        res = x
        x = self.IN(x)
        x = self.conv33(self.conv11(x))
        x = self.act(x)
        x = self.ssd(x)
        x = self.conv111(x)
        return x + res

from einops import rearrange
def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')


def to_4d(x, h, w):
    return rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)




# class SSD5(nn.Module):
#     def __init__(self,dim1,dim2):
#         super().__init__()
#         self.conv1 = nn.Conv2d(dim1,dim2,3,1,1)
#         self.ssd = SpectralGatingNetwork(dim2)
#
#     def forward(self,x):
#         b,c,h,w = x.size()
#         res = x
#         x = self.conv1(x)
#         x = to_3d(x)
#         x = self.ssd(x)
#         x = to_4d(x,h,w)
#         return x + res
########################################

# class SpatialGate(nn.Module):
#     """ Spatial-Gate.
#     Args:
#         dim (int): Half of input channels.
#     """
#     def __init__(self, dim):
#         super().__init__()
#         self.norm = nn.LayerNorm(dim)
#         self.conv = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim) # DW Conv
#
#     def forward(self, x, H, W):
#         # Split
#         x1, x2 = x.chunk(2, dim = -1)
#         B, N, C = x.shape
#         x2 = self.conv(self.norm(x2).transpose(1, 2).contiguous().view(B, C//2, H, W)).flatten(2).transpose(-1, -2).contiguous()
#
#         return x1 * x2

# class Juan1(nn.Module):
#     def __init__(self,dim1,dim2):
#         super().__init__()
#         self.linear1 = nn.Linear(dim1,dim2)
#         self.act = nn.GELU()
#         self.sg = SpatialGate(dim2)
#         self.linear2 = nn.Linear(dim2,dim2)
#
#     def forward(self,x):
#         x = self.linear1(x)
#         x = self.act(x)
#         x = self.sg(x)
#         x = self.linear2(x)
#         return x

class Juan2(nn.Module):
    def __init__(self,dim1,dim2):
        super().__init__()
        self.conv = nn.Conv2d(dim1,dim2,3,1,1)
        self.conv1 = conv1x1(dim1//2,dim2//2)
        self.conv2 = nn.Conv2d(dim1//2,dim2//2,3,1,1)
        self.conv3 = nn.Conv2d(dim1//2,dim2//2,5,1,2)
        self.conv4 = nn.Conv2d(2*dim2,dim2,1)
        self.act = nn.ReLU()
        self.IN = nn.InstanceNorm2d(dim2)

    def forward(self,x):
        res = x
        res2 = self.act(self.IN(self.conv(res)))
        x1,x2 = x.chunk(2,dim=1)
        x3,x4 = x.chunk(2,dim=1)
        x1 = self.act(self.conv1(x1))
        x2 = self.act(self.conv2(x2))
        x3 = self.act(self.conv3(x3))
        x = torch.cat((x1,x2,x3,x4),dim=1)
        x = self.conv4(x)
        return x + res2+res

def conv_init(conv):
    if conv.weight is not None:
        nn.init.kaiming_normal_(conv.weight, mode='fan_out')
    if conv.bias is not None:
        nn.init.constant_(conv.bias, 0)
# 论文：Channel-wise Topology Refinement Graph Convolution for Skeleton-Based Action Recognition
# 论文地址：https://ieeexplore.ieee.org/document/9710007

class Juan3(nn.Module):
    def __init__(self, in_channels, out_channels, rel_reduction=8, mid_reduction=1):
        super(Juan3, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        if in_channels == 3 or in_channels == 9:
            self.rel_channels = 8
            self.mid_channels = 16
        else:
            self.rel_channels = in_channels // rel_reduction
            self.mid_channels = in_channels // mid_reduction
        self.conv1 = nn.Conv2d(self.in_channels, self.rel_channels, kernel_size=1)
        self.conv2 = nn.Conv2d(self.in_channels, self.rel_channels, kernel_size=1)
        self.conv3 = nn.Conv2d(self.in_channels, self.out_channels, kernel_size=1)
        self.conv4 = nn.Conv2d(self.rel_channels, self.out_channels, kernel_size=1)
        self.tanh = nn.Tanh()
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                conv_init(m)

    def forward(self, x, A=None, alpha=1):
        x1, x2, x3 = self.conv1(x).mean(-2), self.conv2(x).mean(-2), self.conv3(x)
        x1 = self.tanh(x1.unsqueeze(-1) - x2.unsqueeze(-2))
        x1 = self.conv4(x1) * alpha + (A.unsqueeze(0).unsqueeze(0) if A is not None else 0)  # N,C,V,V
        x1 = torch.einsum('ncuv,nctv->nctu', x1, x3)
        return x1

from einops.layers.torch import Rearrange

# 论文：DEA-Net: Single image dehazing based on detail-enhanced convolution and content-guided attention
# Github地址：https://github.com/cecret3350/DEA-Net

class Conv2d_cd(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1,
                 padding=1, dilation=1, groups=1, bias=False, theta=1.0):
        super(Conv2d_cd, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding,
                              dilation=dilation, groups=groups, bias=bias)
        self.theta = theta

    def get_weight(self):
        conv_weight = self.conv.weight
        conv_shape = conv_weight.shape
        conv_weight = Rearrange('c_in c_out k1 k2 -> c_in c_out (k1 k2)')(conv_weight)
        conv_weight_cd = torch.cuda.FloatTensor(conv_shape[0], conv_shape[1], 3 * 3).fill_(0)
        conv_weight_cd[:, :, :] = conv_weight[:, :, :]
        conv_weight_cd[:, :, 4] = conv_weight[:, :, 4] - conv_weight[:, :, :].sum(2)
        conv_weight_cd = Rearrange('c_in c_out (k1 k2) -> c_in c_out k1 k2', k1=conv_shape[2], k2=conv_shape[3])(
            conv_weight_cd)
        return conv_weight_cd, self.conv.bias


class Conv2d_ad(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1,
                 padding=1, dilation=1, groups=1, bias=False, theta=1.0):
        super(Conv2d_ad, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding,
                              dilation=dilation, groups=groups, bias=bias)
        self.theta = theta

    def get_weight(self):
        conv_weight = self.conv.weight
        conv_shape = conv_weight.shape
        conv_weight = Rearrange('c_in c_out k1 k2 -> c_in c_out (k1 k2)')(conv_weight)
        conv_weight_ad = conv_weight - self.theta * conv_weight[:, :, [3, 0, 1, 6, 4, 2, 7, 8, 5]]
        conv_weight_ad = Rearrange('c_in c_out (k1 k2) -> c_in c_out k1 k2', k1=conv_shape[2], k2=conv_shape[3])(
            conv_weight_ad)
        return conv_weight_ad, self.conv.bias


class Conv2d_rd(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1,
                 padding=2, dilation=1, groups=1, bias=False, theta=1.0):

        super(Conv2d_rd, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding,
                              dilation=dilation, groups=groups, bias=bias)
        self.theta = theta

    def forward(self, x):

        if math.fabs(self.theta - 0.0) < 1e-8:
            out_normal = self.conv(x)
            return out_normal
        else:
            conv_weight = self.conv.weight
            conv_shape = conv_weight.shape
            if conv_weight.is_cuda:
                conv_weight_rd = torch.cuda.FloatTensor(conv_shape[0], conv_shape[1], 5 * 5).fill_(0)
            else:
                conv_weight_rd = torch.zeros(conv_shape[0], conv_shape[1], 5 * 5)
            conv_weight = Rearrange('c_in c_out k1 k2 -> c_in c_out (k1 k2)')(conv_weight)
            conv_weight_rd[:, :, [0, 2, 4, 10, 14, 20, 22, 24]] = conv_weight[:, :, 1:]
            conv_weight_rd[:, :, [6, 7, 8, 11, 13, 16, 17, 18]] = -conv_weight[:, :, 1:] * self.theta
            conv_weight_rd[:, :, 12] = conv_weight[:, :, 0] * (1 - self.theta)
            conv_weight_rd = conv_weight_rd.view(conv_shape[0], conv_shape[1], 5, 5)
            out_diff = nn.functional.conv2d(input=x, weight=conv_weight_rd, bias=self.conv.bias,
                                            stride=self.conv.stride, padding=self.conv.padding, groups=self.conv.groups)

            return out_diff


class Conv2d_hd(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1,
                 padding=1, dilation=1, groups=1, bias=False, theta=1.0):
        super(Conv2d_hd, self).__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding,
                              dilation=dilation, groups=groups, bias=bias)

    def get_weight(self):
        conv_weight = self.conv.weight
        conv_shape = conv_weight.shape
        conv_weight_hd = torch.cuda.FloatTensor(conv_shape[0], conv_shape[1], 3 * 3).fill_(0)
        conv_weight_hd[:, :, [0, 3, 6]] = conv_weight[:, :, :]
        conv_weight_hd[:, :, [2, 5, 8]] = -conv_weight[:, :, :]
        conv_weight_hd = Rearrange('c_in c_out (k1 k2) -> c_in c_out k1 k2', k1=conv_shape[2], k2=conv_shape[2])(
            conv_weight_hd)
        return conv_weight_hd, self.conv.bias


class Conv2d_vd(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1,
                 padding=1, dilation=1, groups=1, bias=False):
        super(Conv2d_vd, self).__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding,
                              dilation=dilation, groups=groups, bias=bias)

    def get_weight(self):
        conv_weight = self.conv.weight
        conv_shape = conv_weight.shape
        conv_weight_vd = torch.cuda.FloatTensor(conv_shape[0], conv_shape[1], 3 * 3).fill_(0)
        conv_weight_vd[:, :, [0, 1, 2]] = conv_weight[:, :, :]
        conv_weight_vd[:, :, [6, 7, 8]] = -conv_weight[:, :, :]
        conv_weight_vd = Rearrange('c_in c_out (k1 k2) -> c_in c_out k1 k2', k1=conv_shape[2], k2=conv_shape[2])(
            conv_weight_vd)
        return conv_weight_vd, self.conv.bias



class DEC(nn.Module):
    def __init__(self, dim1,dim2):
        super(DEC, self).__init__()
        self.conv1_1 = Conv2d_cd(dim1, dim2, 3, bias=True)
        self.conv1_2 = Conv2d_hd(dim1, dim2, 3, bias=True)
        self.conv1_3 = Conv2d_vd(dim1, dim2, 3, bias=True)
        self.conv1_4 = Conv2d_ad(dim1, dim2, 3, bias=True)
        self.conv1_5 = nn.Conv2d(dim1, dim2, 3, padding=1, bias=True)

    def forward(self, x):
        w1, b1 = self.conv1_1.get_weight()
        w2, b2 = self.conv1_2.get_weight()
        w3, b3 = self.conv1_3.get_weight()
        w4, b4 = self.conv1_4.get_weight()
        w5, b5 = self.conv1_5.weight, self.conv1_5.bias

        w = w1 + w2 + w3 + w4 + w5
        b = b1 + b2 + b3 + b4 + b5
        res = nn.functional.conv2d(input=x, weight=w, bias=b, stride=1, padding=1, groups=1)

        return res



class Juan5(nn.Module):
    def __init__(self,dim1,dim2):
        super().__init__()
        self.conv =  torch.nn.Sequential(
            torch.nn.ReflectionPad2d(1),
            torch.nn.Conv2d(dim1, dim2, 3, 1, 0),
            torch.nn.InstanceNorm2d(dim2),
            torch.nn.ReLU()
        )
        self.rfa = DEC(dim2,dim2)

    def forward(self,x):
        # b,c,h,w = x.size()
        res = x
        res2 = self.conv(res)
        # x = to_3d(x)
        x = self.rfa(x)
        # x = to_4d(x,h,w)
        return x + res2+res



# class CTF(nn.Module):
#     def __init__(self,dim1,dim2):
#         super().__init__()
#
#
#         self.conv =  torch.nn.Sequential(
#             torch.nn.ReflectionPad2d(1),
#             torch.nn.Conv2d(dim1, dim2, 3, 1, 0),
#             torch.nn.InstanceNorm2d(dim2),
#             torch.nn.ReLU()
#         )
#
#
#         self.mamba =  torch.nn.Sequential(
#             torch.nn.ReflectionPad2d(1),
#             torch.nn.Conv2d(dim1, dim2, 3, 1, 0),
#             torch.nn.InstanceNorm2d(dim2),
#             torch.nn.ReLU(),
#         )
#
#         self.fu = fu(dim2*2)
#         self.split_index = (dim2,dim2)
#
#     def forward(self,x1,x2):
#         x1 = self.conv(x1)
#         x2 = self.mamba(x2)
#         x = torch.cat((x1,x2),dim=1)
#         x = self.fu(x)+x
#         x1,x2 = torch.split(x,self.split_index,dim=1)
#         return x1,x2

class CTF(nn.Module):
    def __init__(self,dim1,dim2):
        super().__init__()


        self.conv =  torch.nn.Sequential(
            torch.nn.ReflectionPad2d(1),
            torch.nn.Conv2d(dim1, dim2, 3, 1, 0),
            torch.nn.InstanceNorm2d(dim2),
            torch.nn.ReLU()
        )
        # self.conv = Juan6(dim1,dim2)

        # self.mamba =  torch.nn.Sequential(
        #     torch.nn.ReflectionPad2d(1),
        #     torch.nn.Conv2d(dim1, dim2, 3, 1, 0),
        #     torch.nn.InstanceNorm2d(dim2),
        #     torch.nn.ReLU()
        # )
        self.mamba = SSD2(dim1,dim2)

        self.fu = fu(dim2*2)
        self.split_index = (dim2,dim2)

    def forward(self,x1,x2):
        x1 = self.conv(x1)
        x2 = self.mamba(x2)
        x = torch.cat((x1,x2),dim=1)
        x = self.fu(x)+x
        x1,x2 = torch.split(x,self.split_index,dim=1)
        return x1,x2



# class Backbone(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.conv1 = conv1x1(3,16)
#         self.convma1 = CTF(16,16)
#         self.convma2 = CTF(16,16)
#         self.convma3 = CTF(16,16)
#         self.convma4 = CTF(16,16)
#
#         self.conv2 = conv1x1(32,3)
#
#         self.act = nn.Sigmoid()
#
#     def forward(self,x):
#         x = self.conv1(x)
#         x1, x2 = self.convma1(x,x)
#         x1, x2 = self.convma2(x1,x2)
#         x1, x2 = self.convma3(x1, x2)
#         x1, x2 = self.convma4(x1,x2)
#
#         x = torch.cat((x1,x2),dim=1)
#
#         x = self.act(self.conv2(x))
#
#         return x


__all__ = ['MCALayer', 'MCAGate']


class StdPool(nn.Module):
    def __init__(self):
        super(StdPool, self).__init__()

    def forward(self, x):
        b, c, _, _ = x.size()

        std = x.view(b, c, -1).std(dim=2, keepdim=True)
        std = std.reshape(b, c, 1, 1)

        return std


class MCAGate(nn.Module):
    def __init__(self, k_size, pool_types=['avg', 'std']):
        """Constructs a MCAGate module.
        Args:
            k_size: kernel size
            pool_types: pooling type. 'avg': average pooling, 'max': max pooling, 'std': standard deviation pooling.
        """
        super(MCAGate, self).__init__()

        self.pools = nn.ModuleList([])
        for pool_type in pool_types:
            if pool_type == 'avg':
                self.pools.append(nn.AdaptiveAvgPool2d(1))
            elif pool_type == 'max':
                self.pools.append(nn.AdaptiveMaxPool2d(1))
            elif pool_type == 'std':
                self.pools.append(StdPool())
            else:
                raise NotImplementedError

        self.conv = nn.Conv2d(1, 1, kernel_size=(1, k_size), stride=1, padding=(0, (k_size - 1) // 2), bias=False)
        self.sigmoid = nn.Sigmoid()

        self.weight = nn.Parameter(torch.rand(2))

    def forward(self, x):
        feats = [pool(x) for pool in self.pools]

        if len(feats) == 1:
            out = feats[0]
        elif len(feats) == 2:
            weight = torch.sigmoid(self.weight)
            out = 1 / 2 * (feats[0] + feats[1]) + weight[0] * feats[0] + weight[1] * feats[1]
        else:
            assert False, "Feature Extraction Exception!"

        out = out.permute(0, 3, 2, 1).contiguous()
        out = self.conv(out)
        out = out.permute(0, 3, 2, 1).contiguous()

        out = self.sigmoid(out)
        out = out.expand_as(x)

        return x * out


class MCALayer(nn.Module):
    def __init__(self, inp, no_spatial=False):
        """Constructs a MCA module.
        Args:
            inp: Number of channels of the input feature maps
            no_spatial: whether to build channel dimension interactions
        """
        super(MCALayer, self).__init__()

        lambd = 1.5
        gamma = 1
        temp = round(abs((math.log2(inp) - gamma) / lambd))
        kernel = temp if temp % 2 else temp - 1

        self.h_cw = MCAGate(3)
        self.w_hc = MCAGate(3)
        self.no_spatial = no_spatial
        if not no_spatial:
            self.c_hw = MCAGate(kernel)

    def forward(self, x):
        x_h = x.permute(0, 2, 1, 3).contiguous()
        x_h = self.h_cw(x_h)
        x_h = x_h.permute(0, 2, 1, 3).contiguous()

        x_w = x.permute(0, 3, 2, 1).contiguous()
        x_w = self.w_hc(x_w)
        x_w = x_w.permute(0, 3, 2, 1).contiguous()

        if not self.no_spatial:
            x_c = self.c_hw(x)
            x_out = 1 / 3 * (x_c + x_h + x_w)
        else:
            x_out = 1 / 2 * (x_h + x_w)

        return x_out


class Backbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = conv1x1(3, 16)

        self.convma1 = CTF(16, 16)
        self.convma2 = CTF(16, 16)
        self.convma3 = CTF(16, 16)
        self.convma4 = CTF(16, 16)

        self.mul =  MCALayer(32)     ##############w/o mul
        self.conv2 = conv1x1(32, 3)

        self.act = nn.Sigmoid()

    def forward(self, x):
        res = x
        x = self.conv1(res)

        x1, x2 = self.convma1(x, x)
        x3, x4 = self.convma2(x1, x2)
        x5, x6 = self.convma3(x1+x3, x2+x4)
        x7, x8 = self.convma4(x1+x3+x5, x2+x4+x6)

        # res2 = self.mul(res)
        x = torch.cat((x7, x8), dim=1)

        x = self.act(self.conv2(self.mul(x)))
        # x = self.act(self.conv2(x))   #########w/o mul

        return x


# class Backbone(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.conv1 = conv1x1(3, 16)
#
#         self.convma1 = CTF(16, 16)
#         self.convma2 = CTF(16, 16)
#         self.convma3 = CTF(16, 16)
#         self.convma4 = CTF(16, 16)
#
#         # self.mul =  FIVE_APLUSNet(3)
#         self.conv2 = conv1x1(32, 3)
#
#         self.act = nn.Sigmoid()
#
#     def forward(self, x):
#         res = x
#         x = self.conv1(res)
#
#         x1, x2 = self.convma1(x, x)
#         x3, x4 = self.convma2(x1, x2)
#         x5, x6 = self.convma3(x1+x3, x2+x4)
#         x7, x8 = self.convma4(x1+x3+x5, x2+x4+x6)
#
#         # res2 = self.mul(res)
#         x = torch.cat((x7, x8), dim=1)
#
#         x = self.act(self.conv2(x))
#
#         return x
################################################################################################################################################






from thop import profile
if __name__ == '__main__':
    data = torch.randn([1, 3, 256, 256]).cuda()
    model = Backbone().cuda()
    out = model(data)
    print(out.shape)
    flops, params = profile(model, (data, ))
    print("flops: ", flops / 1e9, "params: ", params / 1e6)
