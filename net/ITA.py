
import torch
import torch.nn as nn
from torch.nn import Parameter
# from switchable_norm import SwitchNorm2d
from torchvision.models.vgg import vgg16
from torch.distributions import kl
# from utils1 import  ResBlock, ConvBlock, Up, Compute_z, PixelShuffleUpsample
import torch.nn.functional as F
from timm.models.layers import trunc_normal_, DropPath


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class swish(nn.Module):
    def __init__(self):
        super(swish, self).__init__()

    def forward(self, x):
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

class Mish(torch.nn.Module):
    def __init__(self):
        super(Mish, self).__init__()

    def forward(self, x):
        return x * torch.tanh(torch.nn.functional.softplus(x))


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
                Mish(),
                nn.Conv2d(_make_divisible(inp // reduction), oup, 1, 1, 0),
                nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x)
        y = self.fc(y).view(b, c, 1, 1).cuda()
        return x * y



# class TNet(torch.nn.Module):  ## 加宽加深的
#     def __init__(self):
#         super().__init__()
#         self.conv1 = torch.nn.Sequential(
#             # torch.nn.ReflectionPad2d(1),
#             torch.nn.Conv2d(3, 64, 1, 1, 0),
#             # SwitchNorm2d(64, using_moving_average=False, using_bn=True),
#             torch.nn.InstanceNorm2d(64),
#             # torch.nn.ReLU()
#             Mish()
#         )
#         self.conv2 = torch.nn.Sequential(
#             torch.nn.ReflectionPad2d(1),
#             torch.nn.Conv2d(64, 256, 3, 1, 0),
#             torch.nn.InstanceNorm2d(256),
#             # SwitchNorm2d(256, using_moving_average=False, using_bn=True),
#             Mish()
#         )
#         self.conv3 = torch.nn.Sequential(
#             torch.nn.ReflectionPad2d(1),
#             torch.nn.Conv2d(256, 256, 3, 1, 0),
#             torch.nn.InstanceNorm2d(256),
#             # SwitchNorm2d(256, using_moving_average=False, using_bn=True),
#             Mish()
#         )
#         self.conv4 = torch.nn.Sequential(
#             torch.nn.ReflectionPad2d(1),
#             torch.nn.Conv2d(256, 256, 3, 1, 0),
#             torch.nn.InstanceNorm2d(256),
#             # SwitchNorm2d(256, using_moving_average=False, using_bn=True),
#             Mish()
#         )
#         self.conv5 = torch.nn.Sequential(
#             torch.nn.ReflectionPad2d(1),
#             torch.nn.Conv2d(256, 64, 3, 1, 0),
#             torch.nn.InstanceNorm2d(64),
#             # SwitchNorm2d(64, using_moving_average=False, using_bn=True),
#             Mish(),
#             SELayer(64, 64)
#         )
#         self.final = torch.nn.Sequential(
#             torch.nn.Conv2d(64, 3, 1, 1, 0),
#             torch.nn.Sigmoid()
#         )
#
#     def forward(self, data):
#         data = self.conv1(data)
#         data = self.conv2(data)
#         data = self.conv3(data)
#         data = self.conv4(data)
#         data = self.conv5(data)
#         data1 = self.final(data)
#
#         return data1


# class TBNet(torch.nn.Module):  ## 加宽加深的
#     def __init__(self):
#         super().__init__()
#         self.conv1 = torch.nn.Sequential(
#             # torch.nn.ReflectionPad2d(1),
#             torch.nn.Conv2d(3, 64, 1, 1, 0),
#             # SwitchNorm2d(64, using_moving_average=False, using_bn=True),
#             torch.nn.InstanceNorm2d(64),
#             # torch.nn.ReLU()
#             Mish()
#         )
#         self.conv2 = torch.nn.Sequential(
#             torch.nn.ReflectionPad2d(1),
#             torch.nn.Conv2d(64, 256, 3, 1, 0),
#             torch.nn.InstanceNorm2d(256),
#             # SwitchNorm2d(256, using_moving_average=False, using_bn=True),
#             Mish()
#         )
#         self.conv3 = torch.nn.Sequential(
#             torch.nn.ReflectionPad2d(1),
#             torch.nn.Conv2d(256, 256, 3, 1, 0),
#             torch.nn.InstanceNorm2d(256),
#             # SwitchNorm2d(256, using_moving_average=False, using_bn=True),
#             Mish()
#         )
#         self.conv4 = torch.nn.Sequential(
#             torch.nn.ReflectionPad2d(1),
#             torch.nn.Conv2d(256, 256, 3, 1, 0),
#             torch.nn.InstanceNorm2d(256),
#             # SwitchNorm2d(256, using_moving_average=False, using_bn=True),
#             Mish()
#         )
#         self.conv5 = torch.nn.Sequential(
#             torch.nn.ReflectionPad2d(1),
#             torch.nn.Conv2d(256, 64, 3, 1, 0),
#             torch.nn.InstanceNorm2d(64),
#             # SwitchNorm2d(64, using_moving_average=False, using_bn=True),
#             Mish(),
#             SELayer(64, 64)
#         )
#         self.final = torch.nn.Sequential(
#             torch.nn.Conv2d(64, 3, 1, 1, 0),
#             torch.nn.Sigmoid()
#         )
#
#     def forward(self, data):
#         data = self.conv1(data)
#         data = self.conv2(data)
#         data = self.conv3(data)
#         data = self.conv4(data)
#         data = self.conv5(data)
#         data1 = self.final(data)
#
#         return data1

# class JNet(torch.nn.Module):  ## 加宽加深的
#     def __init__(self):
#         super().__init__()
#         self.conv1 = torch.nn.Sequential(
#             # torch.nn.ReflectionPad2d(1),
#             torch.nn.Conv2d(3, 64, 1, 1, 0),
#             # SwitchNorm2d(64, using_moving_average=False, using_bn=True),
#             torch.nn.InstanceNorm2d(64),
#             # torch.nn.ReLU()
#             Mish()
#         )
#         self.conv2 = torch.nn.Sequential(
#             torch.nn.ReflectionPad2d(1),
#             torch.nn.Conv2d(64, 256, 3, 1, 0),
#             torch.nn.InstanceNorm2d(256),
#             # SwitchNorm2d(256, using_moving_average=False, using_bn=True),
#             Mish()
#         )
#         self.conv3 = torch.nn.Sequential(
#             torch.nn.ReflectionPad2d(1),
#             torch.nn.Conv2d(256, 256, 3, 1, 0),
#             torch.nn.InstanceNorm2d(256),
#             # SwitchNorm2d(256, using_moving_average=False, using_bn=True),
#             Mish()
#         )
#         self.conv4 = torch.nn.Sequential(
#             torch.nn.ReflectionPad2d(1),
#             torch.nn.Conv2d(256, 256, 3, 1, 0),
#             torch.nn.InstanceNorm2d(256),
#             # SwitchNorm2d(256, using_moving_average=False, using_bn=True),
#             Mish()
#         )
#         self.conv5 = torch.nn.Sequential(
#             torch.nn.ReflectionPad2d(1),
#             torch.nn.Conv2d(256, 64, 3, 1, 0),
#             torch.nn.InstanceNorm2d(64),
#             # SwitchNorm2d(64, using_moving_average=False, using_bn=True),
#             Mish(),
#             SELayer(64, 64)
#         )
#         self.final = torch.nn.Sequential(
#             torch.nn.Conv2d(64, 3, 1, 1, 0),
#             torch.nn.Sigmoid()
#         )
#
#     def forward(self, data):
#         data = self.conv1(data)
#         data = self.conv2(data)
#         data = self.conv3(data)
#         data = self.conv4(data)
#         data = self.conv5(data)
#         data1 = self.final(data)
#
#         return data1

class GNet(nn.Module):  ## 用于估算卷积核
    def __init__(self):
        super(GNet, self).__init__()
        # 定义生成 g_out 的卷积层，可以根据需要调整
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.fc = nn.Linear(64, 3 * 3 * 9 * 9)  # 全连接层，用于输出期望形状

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = torch.relu(self.conv3(x))
        x = x.mean([2, 3])
        x = self.fc(x)
        x = torch.softmax(x, dim=1)
        g_out = x.view(3, 3, 9, 9)
        return g_out

#####################################################################HOUMIAN################################################################################

# if __name__ == "__main__":
#     model = MambaLayer(dim=3).cuda()
#     input = torch.zeros((2, 1, 128, 128)).cuda()  ##
#     input1 = torch.randn(1, 3, 256, 256).cuda()
#     output = model(input1)
#     print(output.shape)

from torch import Tensor
import torch
import torch.nn as nn

class EnhancedPartialConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, n_div=2, bias=False, method='split_cat'):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.n_div = n_div
        self.method = method

        if method not in ['slicing', 'split_cat']:
            raise ValueError("Unsupported forward method. Use 'slicing' or 'split_cat'.")

        # 计算部分卷积通道和未处理通道
        self.dim_conv = in_channels // n_div
        self.dim_untouched = in_channels - self.dim_conv

        # 定义所需卷积层
        self.partial_conv3 = nn.Conv2d(self.dim_conv, self.dim_conv, 3, 1, 1, bias=False)
        self.untouched_conv = nn.Conv2d(self.dim_untouched, out_channels, 3, 1, 1, bias=False)
        self.conv_1x1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=bias)
        self.conv = nn.Conv2d(self.dim_conv, out_channels, kernel_size, stride, padding, bias=bias)

    def forward(self, x: Tensor) -> Tensor:
        if self.method == 'slicing':
            return self.forward_slicing(x)
        elif self.method == 'split_cat':
            return self.forward_split_cat(x)

    def forward_slicing(self, x: Tensor) -> Tensor:
        # 对未处理的部分进行卷积
        untouched_output = self.untouched_conv(x[:, self.dim_conv:, :, :])
        # 对部分卷积通道进行卷积
        conv_output = self.conv(x[:, :self.dim_conv, :, :])
        # 合并结果
        output = torch.add(conv_output, untouched_output)
        return output

    def forward_split_cat(self, x: Tensor) -> Tensor:
        # 通道分割
        x1, x2 = torch.split(x, [self.dim_conv, self.dim_untouched], dim=1)
        # 部分卷积
        x1 = self.partial_conv3(x1)
        # 合并通道并进行 1x1 卷积
        x = torch.cat((x1, x2), dim=1)
        x = self.conv_1x1(x)
        return x


class JNet(torch.nn.Module):
    def __init__(self, num=64):
        super().__init__()
        self.conv1 = torch.nn.Sequential(
            torch.nn.ReflectionPad2d(1),
            torch.nn.Conv2d(3, num, 3, 1, 0),
            torch.nn.InstanceNorm2d(num),
            torch.nn.ReLU()
        )
        self.conv2 = torch.nn.Sequential(
            torch.nn.ReflectionPad2d(1),
            torch.nn.Conv2d(num, num, 3, 1, 0),
            torch.nn.InstanceNorm2d(num),
            torch.nn.ReLU()
        )
        self.conv3 = torch.nn.Sequential(
            torch.nn.ReflectionPad2d(1),
            torch.nn.Conv2d(num, num, 3, 1, 0),
            torch.nn.InstanceNorm2d(num),
            torch.nn.ReLU()
        )
        # self.conv3 = torch.nn.Sequential(
        #     torch.nn.ReflectionPad2d(1),
        #     EnhancedPartialConv2d(num, num, 3, 1, 0, 4, False, method='split_cat'),
        #     torch.nn.InstanceNorm2d(num),
        #     torch.nn.ReLU()
        # )
        # self.conv3 = EnhancedPartialConv2d(num, num, 3, 1, 0, 4, False, method='split_cat'),
        self.conv4 = torch.nn.Sequential(
            torch.nn.ReflectionPad2d(1),
            torch.nn.Conv2d(num, num, 3, 1, 0),
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





class TNet(torch.nn.Module):
    def __init__(self, num=64):
        super().__init__()
        self.conv1 = torch.nn.Sequential(
            torch.nn.ReflectionPad2d(1),
            torch.nn.Conv2d(3, num, 3, 1, 0),
            torch.nn.InstanceNorm2d(num),
            torch.nn.ReLU()
        )
        self.conv2 = torch.nn.Sequential(
            torch.nn.ReflectionPad2d(1),
            torch.nn.Conv2d(num, num, 3, 1, 0),
            torch.nn.InstanceNorm2d(num),
            torch.nn.ReLU()
        )
        self.conv3 = torch.nn.Sequential(
            torch.nn.ReflectionPad2d(1),
            torch.nn.Conv2d(num, num, 3, 1, 0),
            torch.nn.InstanceNorm2d(num),
            torch.nn.ReLU()
        )
        self.conv4 = torch.nn.Sequential(
            torch.nn.ReflectionPad2d(1),
            torch.nn.Conv2d(num, num, 3, 1, 0),
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

class TBNet(torch.nn.Module):
    def __init__(self, num=64):
        super().__init__()
        self.conv1 = torch.nn.Sequential(
            torch.nn.ReflectionPad2d(1),
            torch.nn.Conv2d(3, num, 3, 1, 0),
            torch.nn.InstanceNorm2d(num),
            torch.nn.ReLU()
        )
        self.conv2 = torch.nn.Sequential(
            torch.nn.ReflectionPad2d(1),
            torch.nn.Conv2d(num, num, 3, 1, 0),
            torch.nn.InstanceNorm2d(num),
            torch.nn.ReLU()
        )
        self.conv3 = torch.nn.Sequential(
            torch.nn.ReflectionPad2d(1),
            torch.nn.Conv2d(num, num, 3, 1, 0),
            torch.nn.InstanceNorm2d(num),
            torch.nn.ReLU()
        )
        self.conv4 = torch.nn.Sequential(
            torch.nn.ReflectionPad2d(1),
            torch.nn.Conv2d(num, num, 3, 1, 0),
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


class ConvBlock(nn.Module):
    def __init__(self):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv1 = nn.Conv2d(in_channels=64, out_channels=61, kernel_size=3, stride=1, padding=1, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.drop = nn.Dropout2d(p=.20)#用于卷积层之后，正则化操作，将百分之20的概率将某元素置0

    def forward(self, x):
        x, input_x = x
        a = self.relu(self.conv1(self.relu(self.drop(self.conv(self.relu(self.drop(self.conv(x))))))))
        out = torch.cat((a, input_x), 1)
        return (out, input_x)

class UWnet(nn.Module):
    def __init__(self, num_layers=3):
        super(UWnet, self).__init__()
        self.input = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)
        self.output = nn.Conv2d(in_channels=64, out_channels=3, kernel_size=3, stride=1, padding=1, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.blocks = self.StackBlock(ConvBlock, num_layers)

    def StackBlock(self, block, layer_num):
        layers = []
        for _ in range(layer_num):
            layers.append(block())
        return nn.Sequential(*layers)

    def forward(self, x):
        input_x = x
        x1 = self.relu(self.input(x))
        out, _ = self.blocks((x1, input_x))
        out = self.output(out)
        return out
