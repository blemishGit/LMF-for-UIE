import torch
import torch.nn as nn
from mamba_ssm import Mamba


############################################################################################
# Code Implementation of the MambaIR Model
import warnings
warnings.filterwarnings("ignore")
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
from typing import Optional, Callable
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from mamba_ssm.ops.selective_scan_interface import selective_scan_fn, selective_scan_ref
from einops import rearrange, repeat


"""
最近，选择性结构化状态空间模型，特别是改进版本的Mamba，在线性复杂度的远程依赖建模方面表现出了巨大的潜力。
然而，标准Mamba在低级视觉方面仍然面临一定的挑战，例如局部像素遗忘和通道冗余。在这项工作中，我们引入了局部增强和通道注意力来改进普通 Mamba。
通过这种方式，我们利用了局部像素相似性并减少了通道冗余。大量的实验证明了我们方法的优越性。
"""


NEG_INF = -1000000


class ChannelAttention(nn.Module):
    """Channel attention used in RCAN.
    Args:
        num_feat (int): Channel number of intermediate features.
        squeeze_factor (int): Channel squeeze factor. Default: 16.
    """

    def __init__(self, num_feat, squeeze_factor=16):
        super(ChannelAttention, self).__init__()
        self.attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(num_feat, num_feat // squeeze_factor, 1, padding=0),
            nn.ReLU(inplace=True),
            nn.Conv2d(num_feat // squeeze_factor, num_feat, 1, padding=0),
            nn.Sigmoid())

    def forward(self, x):
        y = self.attention(x)
        return x * y


class CAB(nn.Module):
    def __init__(self, num_feat, is_light_sr= False, compress_ratio=3,squeeze_factor=30):
        super(CAB, self).__init__()
        if is_light_sr: # we use depth-wise conv for light-SR to achieve more efficient
            self.cab = nn.Sequential(
                nn.Conv2d(num_feat, num_feat, 3, 1, 1, groups=num_feat),
                ChannelAttention(num_feat, squeeze_factor)
            )
        else: # for classic SR
            self.cab = nn.Sequential(
                nn.Conv2d(num_feat, num_feat // compress_ratio, 3, 1, 1),
                nn.GELU(),
                nn.Conv2d(num_feat // compress_ratio, num_feat, 3, 1, 1),
                ChannelAttention(num_feat, squeeze_factor)
            )

    def forward(self, x):
        return self.cab(x)


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class DynamicPosBias(nn.Module):
    def __init__(self, dim, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.pos_dim = dim // 4
        self.pos_proj = nn.Linear(2, self.pos_dim)
        self.pos1 = nn.Sequential(
            nn.LayerNorm(self.pos_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.pos_dim, self.pos_dim),
        )
        self.pos2 = nn.Sequential(
            nn.LayerNorm(self.pos_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.pos_dim, self.pos_dim)
        )
        self.pos3 = nn.Sequential(
            nn.LayerNorm(self.pos_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.pos_dim, self.num_heads)
        )

    def forward(self, biases):
        pos = self.pos3(self.pos2(self.pos1(self.pos_proj(biases))))
        return pos

    def flops(self, N):
        flops = N * 2 * self.pos_dim
        flops += N * self.pos_dim * self.pos_dim
        flops += N * self.pos_dim * self.pos_dim
        flops += N * self.pos_dim * self.num_heads
        return flops


class Attention(nn.Module):
    r""" Multi-head self attention module with dynamic position bias.

    Args:
        dim (int): Number of input channels.
        num_heads (int): Number of attention heads.
        qkv_bias (bool, optional):  If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set
        attn_drop (float, optional): Dropout ratio of attention weight. Default: 0.0
        proj_drop (float, optional): Dropout ratio of output. Default: 0.0
    """

    def __init__(self, dim, num_heads, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0.,
                 position_bias=True):

        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.position_bias = position_bias
        if self.position_bias:
            self.pos = DynamicPosBias(self.dim // 4, self.num_heads)

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, H, W, mask=None):
        """
        Args:
            x: input features with shape of (num_groups*B, N, C)
            mask: (0/-inf) mask with shape of (num_groups, Gh*Gw, Gh*Gw) or None
            H: height of each group
            W: width of each group
        """
        group_size = (H, W)
        B_, N, C = x.shape
        assert H * W == N
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4).contiguous()
        q, k, v = qkv[0], qkv[1], qkv[2]

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))  # (B_, self.num_heads, N, N), N = H*W

        if self.position_bias:
            # generate mother-set
            position_bias_h = torch.arange(1 - group_size[0], group_size[0], device=attn.device)
            position_bias_w = torch.arange(1 - group_size[1], group_size[1], device=attn.device)
            biases = torch.stack(torch.meshgrid([position_bias_h, position_bias_w]))  # 2, 2Gh-1, 2W2-1
            biases = biases.flatten(1).transpose(0, 1).contiguous().float()  # (2h-1)*(2w-1) 2

            # get pair-wise relative position index for each token inside the window
            coords_h = torch.arange(group_size[0], device=attn.device)
            coords_w = torch.arange(group_size[1], device=attn.device)
            coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Gh, Gw
            coords_flatten = torch.flatten(coords, 1)  # 2, Gh*Gw
            relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Gh*Gw, Gh*Gw
            relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Gh*Gw, Gh*Gw, 2
            relative_coords[:, :, 0] += group_size[0] - 1  # shift to start from 0
            relative_coords[:, :, 1] += group_size[1] - 1
            relative_coords[:, :, 0] *= 2 * group_size[1] - 1
            relative_position_index = relative_coords.sum(-1)  # Gh*Gw, Gh*Gw

            pos = self.pos(biases)  # 2Gh-1 * 2Gw-1, heads
            # select position bias
            relative_position_bias = pos[relative_position_index.view(-1)].view(
                group_size[0] * group_size[1], group_size[0] * group_size[1], -1)  # Gh*Gw,Gh*Gw,nH
            relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Gh*Gw, Gh*Gw
            attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            nP = mask.shape[0]
            attn = attn.view(B_ // nP, nP, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(
                0)  # (B, nP, nHead, N, N)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class SS2D(nn.Module):
    def __init__(
            self,
            d_model,
            d_state=16,
            d_conv=3,
            expand=2.,
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

        self.selective_scan = selective_scan_fn

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

    def forward_core(self, x: torch.Tensor):
        B, C, H, W = x.shape
        L = H * W
        K = 4
        x_hwwh = torch.stack([x.view(B, -1, L), torch.transpose(x, dim0=2, dim1=3).contiguous().view(B, -1, L)], dim=1).view(B, 2, -1, L)
        xs = torch.cat([x_hwwh, torch.flip(x_hwwh, dims=[-1])], dim=1) # (1, 4, 192, 3136)

        x_dbl = torch.einsum("b k d l, k c d -> b k c l", xs.view(B, K, -1, L), self.x_proj_weight)
        dts, Bs, Cs = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=2)
        dts = torch.einsum("b k r l, k d r -> b k d l", dts.view(B, K, -1, L), self.dt_projs_weight)
        xs = xs.float().view(B, -1, L)
        dts = dts.contiguous().float().view(B, -1, L) # (b, k * d, l)
        Bs = Bs.float().view(B, K, -1, L)
        Cs = Cs.float().view(B, K, -1, L) # (b, k, d_state, l)
        Ds = self.Ds.float().view(-1)
        As = -torch.exp(self.A_logs.float()).view(-1, self.d_state)
        dt_projs_bias = self.dt_projs_bias.float().view(-1) # (k * d)
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

    def forward(self, x: torch.Tensor, **kwargs):
        B, H, W, C = x.shape

        xz = self.in_proj(x)
        x, z = xz.chunk(2, dim=-1)

        x = x.permute(0, 3, 1, 2).contiguous()
        x = self.act(self.conv2d(x))
        y1, y2, y3, y4 = self.forward_core(x)
        assert y1.dtype == torch.float32
        y = y1 + y2 + y3 + y4
        y = torch.transpose(y, dim0=1, dim1=2).contiguous().view(B, H, W, -1)
        y = self.out_norm(y)
        y = y * F.silu(z)
        out = self.out_proj(y)
        if self.dropout is not None:
            out = self.dropout(out)
        return out


class VSSBlock(nn.Module):
    def __init__(
            self,
            hidden_dim: int = 0,
            drop_path: float = 0,
            norm_layer: Callable[..., torch.nn.Module] = partial(nn.LayerNorm, eps=1e-6),
            attn_drop_rate: float = 0,
            d_state: int = 16,
            expand: float = 2.,
            is_light_sr: bool = False,
            **kwargs,
    ):
        super().__init__()
        self.ln_1 = norm_layer(hidden_dim)
        self.self_attention = SS2D(d_model=hidden_dim, d_state=d_state,expand=expand,dropout=attn_drop_rate, **kwargs)
        self.drop_path = DropPath(drop_path)
        self.skip_scale= nn.Parameter(torch.ones(hidden_dim))
        self.conv_blk = CAB(hidden_dim,is_light_sr)
        self.ln_2 = nn.LayerNorm(hidden_dim)
        self.skip_scale2 = nn.Parameter(torch.ones(hidden_dim))



    # def forward(self, input, x_size):
    #     # x [B,HW,C]
    #     B, L, C = input.shape
    #     input = input.view(B, *x_size, C).contiguous()  # [B,H,W,C]
    #     x = self.ln_1(input)
    #     x = input*self.skip_scale + self.drop_path(self.self_attention(x))
    #     x = x*self.skip_scale2 + self.conv_blk(self.ln_2(x).permute(0, 3, 1, 2).contiguous()).permute(0, 2, 3, 1).contiguous()
    #     x = x.view(B, -1, C).contiguous()
    #     return x
    def forward(self, input, x_size):
        # 转换输入格式为 NHWC
        B, C, H, W = input.shape  # 输入格式 NCHW
        input = input.permute(0, 2, 3, 1).contiguous()  # 转为 NHWC: [B, H, W, C]

        x = self.ln_1(input)
        x = input * self.skip_scale + self.drop_path(self.self_attention(x))
        x = x * self.skip_scale2 + self.conv_blk(self.ln_2(x).permute(0, 3, 1, 2).contiguous()).permute(0, 2, 3,
                                                                                                        1).contiguous()

        # 转换输出格式为 NCHW
        x = x.permute(0, 3, 1, 2).contiguous()  # 转回 NCHW: [B, C, H, W]
        return x
###############################################################################################
def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution."""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

class Mish(torch.nn.Module):
    def __init__(self):
        super(Mish, self).__init__()

    def forward(self, x):
        return x * torch.tanh(torch.nn.functional.softplus(x))

class MambaLayer(nn.Module):  ## input (1, 3, 256, 256). output (1, 3, 256, 256).
    def __init__(self, dim, d_state=16, d_conv=4, expand=2):
        super().__init__()
        self.dim = dim

        self.nin = nn.Conv2d(dim, dim, 1, 1, 0)
        self.norm = nn.InstanceNorm2d(dim) # LayerNorm
        self.relu = Mish()
        #
        self.nin2 = nn.Conv2d(dim, dim, 1, 1, 0)
        self.norm2 = nn.InstanceNorm2d(dim)
        self.relu2 = Mish()



        self.mamba = Mamba(
            d_model=dim,  # Model dimension d_model
            d_state=d_state,  # SSM state expansion factor
            d_conv=d_conv,  # Local convolution width
            expand=expand  # Block expansion factor
        )

    def forward(self, x):
        B, C = x.shape[:2]
        x = self.nin(x)
        x = self.norm(x)
        x = self.relu(x)
        act_x = x
        # print(x.shape)
        assert C == self.dim
        n_tokens = x.shape[2:].numel()
        img_dims = x.shape[2:]
        x_flat = x.reshape(B, C, n_tokens).transpose(-1, -2)

        # x_mamba = self.mamba(x_flat)

        x_flip_l = torch.flip(x_flat, dims=[2])
        x_flip_c = torch.flip(x_flat, dims=[1])
        x_flip_lc = torch.flip(x_flat, dims=[1,2])
        x_ori = self.mamba(x_flat)
        x_mamba_l = self.mamba(x_flip_l)
        x_mamba_c = self.mamba(x_flip_c)
        x_mamba_lc = self.mamba(x_flip_lc)
        x_ori_l = torch.flip(x_mamba_l, dims=[2])
        x_ori_c = torch.flip(x_mamba_c, dims=[1])
        x_ori_lc = torch.flip(x_mamba_lc, dims=[1,2])
        x_mamba = (x_ori+x_ori_l+x_ori_c+x_ori_lc)/4

        out = x_mamba.transpose(-1, -2).reshape(B, C, *img_dims)
        # act_x = self.relu3(x)
        out += act_x
        out = self.nin2(out)
        out = self.norm2(out)
        out = self.relu2(out)
        return out



class MambaLine(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = conv1x1(3,16)
        self.mamba1 = MambaLayer(16)
        self.conv2 = conv1x1(16, 32)
        self.mamba2 = MambaLayer(32)
        self.vss = VSSBlock(hidden_dim=32, drop_path=0.1, attn_drop_rate=0.1, d_state=16, expand=2.0, is_light_sr=False)
        self.conv3 = conv1x1(32, 16)
        self.mamba3 = MambaLayer(16)
        self.final = torch.nn.Sequential(
            torch.nn.Conv2d(16, 3, 1, 1, 0),
            torch.nn.Sigmoid()
        )


    def forward(self,x):
        x = self.conv1(x)
        x = self.mamba1(x)
        x = self.conv2(x)
        x = self.mamba2(x)
        x = self.vss(x, (x.shape[2], x.shape[3]))
        x = self.conv3(x)
        x = self.mamba3(x)
        x = self.final(x)
        return x
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

##################################################ffcCBAM###################################################################

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

class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1 = nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)

        self.fu = FourierUnit(in_planes, in_planes)

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))

        fu_out = self.fu(x)

        out = avg_out + max_out + fu_out
        return torch.sigmoid(out)

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

class CBAM(nn.Module):
    def __init__(self, in_planes, ratio=16, kernel_size=7):
        super(CBAM, self).__init__()
        self.ca = ChannelAttention(in_planes, ratio)
        self.sa = SpatialAttention(kernel_size)

    def forward(self, x):
        out = self.ca(x) * x
        out = self.sa(out) * out
        return out


import math
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
        self.fft = FourierUnit(inp,inp)

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



        return self.fft(x_out)+x



#######################################################################################################################

class fu(nn.Module):
    def __init__(self,dim):
        super().__init__()
        self.conv1 = nn.Conv2d(dim,dim,1)
        self.conv2 = nn.Conv2d(dim, dim, 1)
        self.ffc = MCALayer(dim)
        self.act = nn.ReLU()

    def forward(self,x):
        res = x
        x = self.conv1(x)
        x = self.ffc(x)
        x = self.act(x)
        x = self.conv2(x)
        return  x + res


class CTF(nn.Module):
    def __init__(self,dim1,dim2):
        super().__init__()


        self.conv =  torch.nn.Sequential(
            torch.nn.ReflectionPad2d(1),
            torch.nn.Conv2d(dim1, dim2, 3, 1, 0),
            torch.nn.InstanceNorm2d(dim2),
            torch.nn.ReLU()
        )

        # self.mamba = torch.nn.Sequential(
        #     conv1x1(dim1, dim2),
        #     SS2D(dim2),
        #     torch.nn.InstanceNorm2d(dim2),
        #     torch.nn.ReLU()
        # )
        self.mamba =  torch.nn.Sequential(
            torch.nn.ReflectionPad2d(1),
            torch.nn.Conv2d(dim1, dim2, 3, 1, 0),
            torch.nn.InstanceNorm2d(dim2),
            torch.nn.ReLU(),
        )


        self.fu = fu(dim2*2)
        self.split_index = (dim2,dim2)

    def forward(self,x1,x2):
        x1 = self.conv(x1)+x1
        x2 = self.mamba(x2)+x2
        x = torch.cat((x1,x2),dim=1)
        x = self.fu(x)+x
        x1,x2 = torch.split(x,self.split_index,dim=1)
        return x1,x2

# class CTF(nn.Module):
#     def __init__(self,dim1,dim2):
#         super().__init__()
#
#         self.fu = fu(dim2*2)
#         self.split_index = (dim2,dim2)
#
#     def forward(self,x1,x2):
#         x = torch.cat((x1,x2),dim=1)
#         x = self.fu(x)+x
#         x1,x2 = torch.split(x,self.split_index,dim=1)
#         return x1,x2


# class backbone(nn.Module):
#     def __init__(self):
#         super().__init__()
#
#         self.conv1 = torch.nn.Sequential(
#             torch.nn.ReflectionPad2d(1),
#             torch.nn.Conv2d(3, 16, 3, 1, 0),
#             torch.nn.InstanceNorm2d(16),
#             torch.nn.ReLU()
#         )
#         self.mamba1 = torch.nn.Sequential(
#             conv1x1(3, 16),
#             MambaLayer(16),
#             torch.nn.InstanceNorm2d(16),
#             torch.nn.ReLU()
#         )
#         self.conv2 = torch.nn.Sequential(
#             torch.nn.ReflectionPad2d(1),
#             torch.nn.Conv2d(16, 16, 3, 1, 0),
#             torch.nn.InstanceNorm2d(16),
#             torch.nn.ReLU()
#         )
#         self.mamba2 = torch.nn.Sequential(
#             conv1x1(16, 16),
#             MambaLayer(16),
#             torch.nn.InstanceNorm2d(16),
#             torch.nn.ReLU()
#         )
#         self.conv3 = torch.nn.Sequential(
#             torch.nn.ReflectionPad2d(1),
#             torch.nn.Conv2d(16, 3, 3, 1, 0),
#             torch.nn.InstanceNorm2d(3),
#             torch.nn.ReLU()
#         )
#         self.mamba3 = torch.nn.Sequential(
#             conv1x1(16, 3),
#             MambaLayer(3),
#             torch.nn.InstanceNorm2d(3),
#             torch.nn.ReLU()
#         )
#         self.fu =  fu(32)
#         self.final = torch.nn.Sequential(
#             torch.nn.Conv2d(3, 3, 1, 1, 0),
#             torch.nn.Sigmoid()
#         )
#
#     def forward(self,x1,x2):
#         x1 = self.conv1(x1)
#         x2 = self.mamba1(x2)
#         x = torch.cat((x1,x2),dim=1)
#         x = self.fu(x)
#



# class Juan(nn.Module):
#     def __init__(self,dim1,dim2):
#         super().__init__()
#         self.conv = torch.nn.Sequential(
#             torch.nn.ReflectionPad2d(1),
#             torch.nn.Conv2d(dim1, dim2, 3, 1, 0),
#             torch.nn.InstanceNorm2d(dim2),
#             torch.nn.ReLU()
#         )
#
#     def forward(self,x):
#         return self.conv(x)

class DWSeparableConv(nn.Module):
    def __init__(self, c1, c2, k=1, s=1, p=0, d=1, g=1):
        super().__init__()
        self.depthwise_h = nn.Conv2d(c1, c1, (1, k), s, padding=(0, p), dilation=(1, d), groups=g)
        self.depthwise_w = nn.Conv2d(c1, c1, (k, 1), s, padding=(p, 0), dilation=(d, 1), groups=g)
        self.pointwise = nn.Conv2d(c1, c2, 1)

    def forward(self, x):
        x = self.depthwise_h(x)
        x = self.depthwise_w(x)
        x = self.pointwise(x)
        return x

class Juan(nn.Module): # 定义一个名为LKSP的神经网络模型类，继承了PyTorch的nn.Module基类。
    def __init__(self, c1, c2, d=[1, 1, 2, 3]): # 初始化函数，需要传入输入信道数(c1), 输出信道数(c2)以及四个不同尺度的卷积核(d=[1, 4, 5, 6])。
        super().__init__() # 调用父类的初始化函数。
        c_ = c1 // 2 # 将输入信道数c1除以2，将结果赋值给变量c_作为隐藏信道。

        # 使用nn.Conv2d模块为网络添加一个2D卷积层，名为aspp1，该层采用1x1的卷积核，扩张率为1，组数为c1和c_的最大公因数。
        # self.aspp1 = nn.Conv2d(c1, c_, 1, 1, padding=0, dilation=1, groups=math.gcd(c1, c_))

        # 下面分别定义带孔卷积层aspp2, aspp3, aspp4，其中扩张率、卷积核尺度和填充padding均不同，实现了多尺度信息的获取。
        self.aspp2 = DWSeparableConv(c1, c_, d[1], 1, p=int(d[1] * (d[1] - 1) / 2), d=d[1])
        self.aspp3 = DWSeparableConv(c1, c_, d[2], 1, p=int(d[2] * (d[2] - 1) / 2), d=d[2])
        self.aspp4 = DWSeparableConv(c1, c_, d[3], 1, p=int(d[3] * (d[3] - 1) / 2), d=d[3])

        # 自适应全局平均池化层，用于获取全局上下文信息，这里的输出尺寸是1x1。
        self.global_avg_pool = nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)),
        nn.Conv2d(c1, c_, 1, stride=1),
        nn.ReLU())

        # 最后的输出卷积层，对各层的结果进行逐通道的汇总，使用BatchNorm进行归一化，使用ReLU激活函数增加非线性，以及使用dropout防止过拟合。
        self.output_conv = nn.Sequential(nn.Conv2d(c_ * 4, c2, 1, bias=False),
        nn.BatchNorm2d(c2),
        nn.ReLU(),
        nn.Dropout(0.5))

    # 前向传播函数，定义了模型的运行流程。
    def forward(self, x):
        # 输入x分别通过aspp1到aspp4以及全局平均池化层，得到5个特征图。
        # x1 = self.aspp1(x)
        x2 = self.aspp2(x)
        x3 = self.aspp3(x)
        x4 = self.aspp4(x)
        x5 = self.global_avg_pool(x)

        # 特征图x5的高和宽通过双线性插值的方式调整至与x4一致。
        x5 = F.interpolate(x5, size=x4.size()[2:], mode='bilinear', align_corners=True)

        # 将这5个特征图在信道维上进行拼接。
        x = torch.cat((x2, x3, x4, x5), dim=1)

        # 通过输出卷积层，得到最终结果。
        return self.output_conv(x)


class Mblock(nn.Module):
    def __init__(self,dim1,dim2):
        super().__init__()
        self.mamba = torch.nn.Sequential(
            conv1x1(dim1, dim2),
            SS2D(dim2),
            torch.nn.InstanceNorm2d(dim2),
            torch.nn.ReLU()
        )
    def forward(self,x):
        return self.mamba(x)


# class Backbone(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.conv1 = conv1x1(3,3)
#         self.juan1 = Juan(3,16)
#         self.mblock1 = Mblock(3,16)
#         self.juan2 = Juan(16,32)
#         self.mblock2 = Mblock(16,32)
#         self.juan3 = Juan(32,16)
#         self.mblock3 = Mblock(32,16)
#         self.juan4 = Juan(16,16)
#         self.mblock4 = Mblock(16,16)
#
#
#         self.convma1 = CTF(3,16)
#         self.convma2 = CTF(16,32)
#         self.convma3 = CTF(32, 16)
#         self.convma4 = CTF(16,16)
#
#
#         self.conv2 = conv1x1(32, 3)
#         self.act = nn.Sigmoid()
#
#     def forward(self,x):
#         x = self.conv1(x)
#
#         x1 = self.juan1(x)
#         x2 = self.mblock1(x)
#         x3,x4 = self.convma1(x1,x2)
#         x1 = x1+x3
#         x2 = x1+x4
#
#         x1 = self.juan2(x1)
#         x2 = self.mblock2(x2)
#         x5,x6 = self.convma2(x1,x2)
#         x1 = x1+x5
#         x2 = x1+x6
#
#         x1 = self.juan3(x1)
#         x2 = self.mblock3(x2)
#         x7,x8 = self.convma3(x1,x2)
#         x1 = x1+x7
#         x2 = x1+x8
#
#         x1 = self.juan4(x1)
#         x2 = self.mblock4(x2)
#         x9,x10 = self.convma4(x1,x2)
#
#         x1 = x1+x9
#         x2 = x1+x10
#
#         x = torch.cat((x1,x2),dim=1)
#         x = self.act(self.conv2(x))
#
#         return x


class Backbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = conv1x1(3,16)
        self.convma1 = CTF(16,16)
        self.convma2 = CTF(16,16)
        self.convma3 = CTF(16, 16)
        self.convma4 = CTF(16,16)
        self.conv2 = conv1x1(32, 3)
        self.act = nn.Sigmoid()

    def forward(self,x):
        x = self.conv1(x)
        x1, x2 = self.convma1(x,x)
        x1, x2 = self.convma2(x1,x2)
        x1, x2 = self.convma3(x1, x2)
        x1, x2 = self.convma4(x1,x2)

        x = torch.cat((x1,x2),dim=1)
        x = self.act(self.conv2(x))

        return x







if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # 检测是否有 GPU
    input = torch.randn(1, 3, 256, 256).to(device)  # 将输入张量移动到 GPU
    wtconv = Backbone().to(device)  # 将模型移动到 GPU
    output = wtconv(input)  # 前向传播
    print(output.size())

