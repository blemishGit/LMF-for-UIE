import torch
import torch.nn as nn


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
        self.ffc = CAM(dim,2)
        self.act = nn.ReLU()

    def forward(self,x):
        res = x
        x = self.conv1(x)
        x = self.ffc(x)
        x = self.act(x)
        x = self.conv2(x)
        return  x + res



################################################################################################################################################

############################################################Conv-Mamba  Jnet########################################################################################################
def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution."""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


####################################################SS2D#####################################################################
class EMA(nn.Module):
    def __init__(self, channels, c2=None, factor=16):
        super(EMA, self).__init__()
        self.groups = factor
        assert channels // self.groups > 0
        self.softmax = nn.Softmax(-1)
        self.agp = nn.AdaptiveAvgPool2d((1, 1))
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))
        self.gn = nn.GroupNorm(channels // self.groups, channels // self.groups)
        self.conv1x1 = nn.Conv2d(channels // self.groups, channels // self.groups, kernel_size=1, stride=1, padding=0)
        self.conv3x3 = nn.Conv2d(channels // self.groups, channels // self.groups, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        #(B,C,H,W)
        b, c, h, w = x.size()

        ### 坐标注意力模块  ###
        group_x = x.reshape(b * self.groups, -1, h, w)  # 在通道方向上将输入分为G组: (B,C,H,W)-->(B*G,C/G,H,W)
        x_h = self.pool_h(group_x) # 使用全局平均池化压缩水平空间方向: (B*G,C/G,H,W)-->(B*G,C/G,H,1)
        x_w = self.pool_w(group_x).permute(0, 1, 3, 2) # 使用全局平均池化压缩垂直空间方向: (B*G,C/G,H,W)-->(B*G,C/G,1,W)-->(B*G,C/G,W,1)
        hw = self.conv1x1(torch.cat([x_h, x_w], dim=2))# 将水平方向和垂直方向的全局特征进行拼接: (B*G,C/G,H+W,1), 然后通过1×1Conv进行变换,来编码空间水平和垂直方向上的特征
        x_h, x_w = torch.split(hw, [h, w], dim=2) # 沿着空间方向将其分割为两个矩阵表示: x_h:(B*G,C/G,H,1); x_w:(B*G,C/G,W,1)

        ### 1×1分支和3×3分支的输出表示  ###
        x1 = self.gn(group_x * x_h.sigmoid() * x_w.permute(0, 1, 3, 2).sigmoid()) # 通过水平方向权重和垂直方向权重调整输入,得到1×1分支的输出: (B*G,C/G,H,W) * (B*G,C/G,H,1) * (B*G,C/G,1,W)=(B*G,C/G,H,W)
        x2 = self.conv3x3(group_x) # 通过3×3卷积提取局部上下文信息: (B*G,C/G,H,W)-->(B*G,C/G,H,W)

        ### 跨空间学习 ###
        ## 1×1分支生成通道描述符来调整3×3分支的输出
        x11 = self.softmax(self.agp(x1).reshape(b * self.groups, -1, 1).permute(0, 2, 1)) # 对1×1分支的输出执行平均池化,然后通过softmax获得归一化后的通道描述符: (B*G,C/G,H,W)-->agp-->(B*G,C/G,1,1)-->reshape-->(B*G,C/G,1)-->permute-->(B*G,1,C/G)
        x12 = x2.reshape(b * self.groups, c // self.groups, -1)  # 将3×3分支的输出进行变换,以便与1×1分支生成的通道描述符进行相乘: (B*G,C/G,H,W)-->reshape-->(B*G,C/G,H*W)
        y1 = torch.matmul(x11, x12) # (B*G,1,C/G) @ (B*G,C/G,H*W) = (B*G,1,H*W)

        ## 3×3分支生成通道描述符来调整1×1分支的输出
        x21 = self.softmax(self.agp(x2).reshape(b * self.groups, -1, 1).permute(0, 2, 1)) # 对3×3分支的输出执行平均池化,然后通过softmax获得归一化后的通道描述符: (B*G,C/G,H,W)-->agp-->(B*G,C/G,1,1)-->reshape-->(B*G,C/G,1)-->permute-->(B*G,1,C/G)
        x22 = x1.reshape(b * self.groups, c // self.groups, -1)  # b*g, c//g, hw  # 将1×1分支的输出进行变换,以便与3×3分支生成的通道描述符进行相乘: (B*G,C/G,H,W)-->reshape-->(B*G,C/G,H*W)
        y2 = torch.matmul(x21, x22)  # (B*G,1,C/G) @ (B*G,C/G,H*W) = (B*G,1,H*W)

        # 聚合两种尺度的空间位置信息, 通过sigmoid生成空间权重, 从而再次调整输入表示
        weights = (y1+y2).reshape(b * self.groups, 1, h, w)  # 将两种尺度下的空间位置信息进行聚合: (B*G,1,H*W)-->reshape-->(B*G,1,H,W)
        weights_ =  weights.sigmoid() # 通过sigmoid生成权重表示: (B*G,1,H,W)
        out = (group_x * weights_).reshape(b, c, h, w) # 通过空间权重再次校准输入: (B*G,C/G,H,W)*(B*G,1,H,W)==(B*G,C/G,H,W)-->reshape(B,C,H,W)
        return out

from mmcv.cnn import build_norm_layer
" pip install mmcv==2.0.0 -f https://download.openmmlab.com/mmcv/dist/cu117/torch1.13/index.html   "

"SEAFORMER: SQUEEZE-ENHANCED AXIAL TRANSFORMER FOR MOBILE SEMANTIC SEGMENTATION"
import torch.nn.functional as F

class Conv2d_BN(nn.Sequential):
    def __init__(self, a, b, ks=1, stride=1, pad=0, dilation=1,
                 groups=1, bn_weight_init=1,
                 norm_cfg=dict(type='BN', requires_grad=True)):
        super().__init__()
        self.inp_channel = a
        self.out_channel = b
        self.ks = ks
        self.pad = pad
        self.stride = stride
        self.dilation = dilation
        self.groups = groups

        self.add_module('c', nn.Conv2d(
            a, b, ks, stride, pad, dilation, groups, bias=False))
        bn = build_norm_layer(norm_cfg, b)[1]
        nn.init.constant_(bn.weight, bn_weight_init)
        nn.init.constant_(bn.bias, 0)
        self.add_module('bn', bn)



class SqueezeAxialPositionalEmbedding(nn.Module):
    def __init__(self, dim, shape):
        super().__init__()

        self.pos_embed = nn.Parameter(torch.randn([1, dim, shape]), requires_grad=True)

    def forward(self, x):
        # (B,C_qk,H)
        B, C, N = x.shape
        x = x + F.interpolate(self.pos_embed, size=(N), mode='linear', align_corners=False)
        return x



class Sea_Attention(torch.nn.Module):
    def __init__(self, dim, key_dim, num_heads,
                 attn_ratio=2,
                 activation=nn.ReLU,
                 norm_cfg=dict(type='BN', requires_grad=True), ):
        super().__init__()
        self.num_heads = num_heads
        self.scale = key_dim ** -0.5
        self.key_dim = key_dim
        self.nh_kd = nh_kd = key_dim * num_heads  # num_head key_dim
        self.d = int(attn_ratio * key_dim)
        self.dh = int(attn_ratio * key_dim) * num_heads
        self.attn_ratio = attn_ratio

        self.to_q = Conv2d_BN(dim, nh_kd, 1, norm_cfg=norm_cfg)
        self.to_k = Conv2d_BN(dim, nh_kd, 1, norm_cfg=norm_cfg)
        self.to_v = Conv2d_BN(dim, self.dh, 1, norm_cfg=norm_cfg)

        self.proj = torch.nn.Sequential(activation(), Conv2d_BN(
            self.dh, dim, bn_weight_init=0, norm_cfg=norm_cfg))
        self.proj_encode_row = torch.nn.Sequential(activation(), Conv2d_BN(
            self.dh, self.dh, bn_weight_init=0, norm_cfg=norm_cfg))
        self.pos_emb_rowq = SqueezeAxialPositionalEmbedding(nh_kd, 16)
        self.pos_emb_rowk = SqueezeAxialPositionalEmbedding(nh_kd, 16)
        self.proj_encode_column = torch.nn.Sequential(activation(), Conv2d_BN(
            self.dh, self.dh, bn_weight_init=0, norm_cfg=norm_cfg))
        self.pos_emb_columnq = SqueezeAxialPositionalEmbedding(nh_kd, 16)
        self.pos_emb_columnk = SqueezeAxialPositionalEmbedding(nh_kd, 16)

        self.dwconv = Conv2d_BN(2 * self.dh, 2 * self.dh, ks=3, stride=1, pad=1, dilation=1,
                                groups=2 * self.dh, norm_cfg=norm_cfg)
        self.act = activation()
        self.pwconv = Conv2d_BN(2 * self.dh, dim, ks=1, norm_cfg=norm_cfg)
        # self.sigmoid = torch.sigmoid()

    def forward(self, x):  # x (B,N,C)
        B, C, H, W = x.shape

        q = self.to_q(x) # 生成query: (B,C,H,W)-->(B,C_qk,H,W)
        k = self.to_k(x) # 生成key: (B,C,H,W)-->(B,C_qk,H,W)
        v = self.to_v(x) # 生成value: (B,C,H,W)-->(B,C_v,H,W)

        # Detail enhancement kernel
        qkv = torch.cat([q, k, v], dim=1) # 将qkv拼接: (B,2*C_qk+C_v,H,W)
        qkv = self.act(self.dwconv(qkv)) # 执行3×3卷积,建模局部空间依赖,从而增强局部细节感知: (B,2*C_qk+C_v,H,W)-->(B,2*C_qk+C_v,H,W)
        qkv = self.pwconv(qkv) # 执行1×1卷积,将通道数量从(2*C_qk+C_v)映射到C,从而生成细节增强特征: (B,2C_qk+C_v,H,W)-->(B,C,H,W)

        # squeeze axial attention
        ## squeeze row, squeeze操作将全局信息保留到单个轴上，然后分别应用自注意力建模对应轴的长期依赖
        qrow = self.pos_emb_rowq(q.mean(-1)).reshape(B, self.num_heads, -1, H).permute(0, 1, 3, 2) #通过平均池化压缩水平方向,并为垂直方向的空间位置添加位置嵌入: (B,C_qk,H,W)-->mean-->(B,C_qk,H)-->reshape-->(B,h,d,H)-->permute-->(B,h,H,d);   C_qk=h*d, h:注意力头的个数；d:每个注意力头的通道数
        krow = self.pos_emb_rowk(k.mean(-1)).reshape(B, self.num_heads, -1, H) #通过平均池化压缩水平方向,并为垂直方向的空间位置添加位置嵌入: (B,C_qk,H,W)-->mean-->(B,C_qk,H)-->reshape-->(B,h,d,H)
        vrow = v.mean(-1).reshape(B, self.num_heads, -1, H).permute(0, 1, 3, 2) #通过平均池化压缩水平方向: (B,C_v,H,W)-->mean-->(B,C_v,H)-->reshape-->(B,h,d_v,H)-->permute-->(B,h,H,d_v);   C_v=h*d_v, h:注意力头的个数；d_v:Value矩阵中每个注意力头的通道数

        attn_row = torch.matmul(qrow, krow) * self.scale  # 计算水平方向压缩之后的自注意力机制：(B,h,H,d) @ (B,h,d,H) = (B,h,H,H)
        attn_row = attn_row.softmax(dim=-1) # 执行softmax操作
        xx_row = torch.matmul(attn_row, vrow)  # 对Value进行加权求和: (B,h,H,H) @ (B,h,H,d_v) = (B,h,H,d_v)
        xx_row = self.proj_encode_row(xx_row.permute(0, 1, 3, 2).reshape(B, self.dh, H, 1)) # 对注意力机制的输出进行reshape操作,并进行卷积：(B,h,H,d_v)-->permute-->(B,h,d_v,H)-->reshape-->(B,C_v,H,1);   C_v=h*d_v

        ## squeeze column
        qcolumn = self.pos_emb_columnq(q.mean(-2)).reshape(B, self.num_heads, -1, W).permute(0, 1, 3, 2) # 通过平均池化压缩垂直方向,并为水平方向的空间位置添加位置嵌入: (B,C_qk,H,W)-->mean-->(B,C_qk,W)-->reshape-->(B,h,d,W)-->permute-->(B,h,W,d);  C_qk=h*d, h:注意力头的个数；d:每个注意力头的通道数
        kcolumn = self.pos_emb_columnk(k.mean(-2)).reshape(B, self.num_heads, -1, W) # 通过平均池化压缩垂直方向,并为水平方向的空间位置添加位置嵌入: (B,C_qk,H,W)-->mean-->(B,C_qk,W)-->reshape-->(B,h,d,W)
        vcolumn = v.mean(-2).reshape(B, self.num_heads, -1, W).permute(0, 1, 3, 2)  #通过平均池化压缩垂直方向: (B,C_v,H,W)-->mean-->(B,C_v,W)-->reshape-->(B,h,d_v,W)-->permute-->(B,h,W,d_v)

        attn_column = torch.matmul(qcolumn, kcolumn) * self.scale # 计算垂直方向压缩之后的自注意力机制：(B,h,W,d) @ (B,h,d,W) = (B,h,W,W)
        attn_column = attn_column.softmax(dim=-1) # 执行softmax操作
        xx_column = torch.matmul(attn_column, vcolumn)  # 对Value进行加权求和: (B,h,W,W) @ (B,h,W,d_v) = (B,h,W,d_v)
        xx_column = self.proj_encode_column(xx_column.permute(0, 1, 3, 2).reshape(B, self.dh, 1, W)) # 对注意力机制的输出进行reshape操作,并进行卷积：(B,h,W,d_v)-->permute-->(B,h,d_v,W)-->reshape-->(B,C_v,1,W);  C_v=h*d_v

        xx = xx_row.add(xx_column) # 将两个注意力机制的输出进行相加,这是一种broadcast操作: (B,C_v,H,1) + (B,C_v,1,W) =(B,C_v,H,W)
        xx = v.add(xx) # 添加残差连接
        xx = self.proj(xx) # 应用1×1Conv得到Squeeze Axial attention的输出
        xx = xx.sigmoid() * qkv  # 为Squeeze Axial attention的输出应用门控机制获得权重, 然后与Detail enhancement kernel的输出进行逐点乘法
        return xx


class CTF(nn.Module):
    def __init__(self,dim1,dim2):
        super().__init__()


        self.conv =  torch.nn.Sequential(
            torch.nn.ReflectionPad2d(1),
            torch.nn.Conv2d(dim1, dim2, 3, 1, 0),
            torch.nn.InstanceNorm2d(dim2),
            torch.nn.ReLU(),
            Sea_Attention(dim2,key_dim=2, num_heads=2)

        )


        self.mamba =  torch.nn.Sequential(
            torch.nn.ReflectionPad2d(1),
            torch.nn.Conv2d(dim1, dim2, 3, 1, 0),
            torch.nn.InstanceNorm2d(dim2),
            torch.nn.ReLU(),
        )

        self.fu = fu(dim2*2)
        self.split_index = (dim2,dim2)

    def forward(self,x1,x2):
        x1 = self.conv(x1)
        x2 = self.mamba(x2)
        x = torch.cat((x1,x2),dim=1)
        x = self.fu(x)+x
        x1,x2 = torch.split(x,self.split_index,dim=1)
        return x1,x2
#
# class CTF(nn.Module):
#     def __init__(self,dim1,dim2):
#         super().__init__()
#
#
#         # self.conv =  torch.nn.Sequential(
#         #     torch.nn.ReflectionPad2d(1),
#         #     torch.nn.Conv2d(dim1, dim2, 3, 1, 0),
#         #     torch.nn.InstanceNorm2d(dim2),
#         #     torch.nn.ReLU()
#         # )
#         self.conv = Juan6(dim1,dim2)
#
#         self.mamba =  torch.nn.Sequential(
#             torch.nn.ReflectionPad2d(1),
#             torch.nn.Conv2d(dim1, dim2, 3, 1, 0),
#             torch.nn.InstanceNorm2d(dim2),
#             torch.nn.ReLU()
#         )
#         # self.mamba = SSD2(dim1,dim2)
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



class Backbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = conv1x1(3,16)
        self.convma1 = CTF(16,16)
        self.convma2 = CTF(16,16)
        self.convma3 = CTF(16,16)
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

################################################################################################################################################






from thop import profile
if __name__ == '__main__':
    data = torch.randn([1, 3, 256, 256]).cuda()
    model = Backbone().cuda()
    out = model(data)
    print(out.shape)
    flops, params = profile(model, (data, ))
    print("flops: ", flops / 1e9, "params: ", params / 1e6)
