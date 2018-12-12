import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import utils
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST

BIT_P_w = 8
BIT_P_b = 32
BIT_F = 8
RANGE_MIN = 0.96
RANGE_MAX = 0.96

def fix(tensor, bit, is_first):
    tmp = tensor.detach()
    tmp_min = torch.min(tmp) * RANGE_MIN if is_first else torch.min(tmp)
    tmp_max = torch.max(tmp) * RANGE_MAX if is_first else torch.max(tmp)
    tmp = torch.clamp(tmp, min = tmp_min, max = tmp_max)
    scale = (tmp_max - tmp_min) / (2**bit - 1)
    zero = tmp_min
    tmp = (torch.round((tmp - tmp_min) /scale)) * scale + tmp_min
    return tmp, scale, zero


class fix_conv2d_block(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                    padding=0, bias=True, bn = False, activation = None, training=True):
        super(fix_conv2d_block, self).__init__()
        assert (bias is not bn)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.activation = activation
        self.training = training

        # fix parameters
        self.scale_F = 0
        self.scale_P_w = 0
        self.sclae_P_b = 0
        self.zero_F = None
        self.zero_P_w = None
        self.zero_P_b = None

        # in the first time, need to deal with outleir
        self.is_first = True

        self.weight = nn.Parameter(torch.Tensor(out_channels, in_channels, kernel_size, kernel_size))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

        self.weight.data.normal_()
        if self.bias is not None:
            self.bias.data.uniform_(-0.1,0.1)

        if bn:
            self.bn = nn.BatchNorm2d(self.out_channels)
        else:
            self.register_parameter('bn', None)

    def forward(self, input):
        self.weight.data, self.scale_P_w, self.zero_P_w = fix(self.weight, bit = BIT_P_w, is_first = self.is_first)
        if self.bias is not None:
            self.bias.data, self.scale_P_b, self.zero_P_b = fix(self.bias, bit = BIT_P_b, is_first = self.is_first)
        output = F.conv2d(input, self.weight, self.bias, self.stride,
                self.padding)

        if self.bn is not None:
            output = self.bn(output)

        if not self.activation is None:
            output = self.activation(output)
            output, self.scale_F, self.zero_F = fix(output, bit = BIT_F, is_first = self.is_first)

        # after forward one time, just set is_first to False
        self.is_first = False
        return output
