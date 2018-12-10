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
RANGE_MIN = 0.95
RANGE_MAX = 0.95

def G_fix(tensor, bit):
    tmp = tensor.detach()
    tmp_min = torch.min(tmp) * RANGE_MIN
    tmp_max = torch.max(tmp) * RANGE_MAX
    tmp = torch.clamp(tmp, min = tmp_min, max = tmp_max)
    delta_r = (tmp_max - tmp_min) / (2**BIT_LEN - 1)
    tmp = (torch.round(tmp - tmp_min) /delta_r) * delta_r + tmp_min
    return tmp, delta_r


class G_fix_Conv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                    padding=0, bias=True, activation = None, training=True):
        super(G_fix_Conv2d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.activation = activation
        self.training = training

        # scale for S
        self.scale_F = 0
        self.scale_P_w = 0
        self.sclae_P_b = 0

        self.weight = nn.Parameter(torch.Tensor(out_channels, in_channels, kernel_size, kernel_size))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

        self.weight.data.normal_()
        if self.bias is not None:
            self.bias.data.uniform_(-0.1,0.1)

    def fix_parameter(self,):
        weight_data, s_w = G_fix(self.weight, bit = BIT_P_w)
        self.weight.data = weight_data
        self.scale_P_w = s_w

        bias_data, s_b = G_fix(self.bias, bit = BIT_P_b)
        self.bias.data = bias_data
        self.scale_P_b = s_b

    def forward(self, input):
        self.fix_parameter()
        output = F.conv2d(input, self.weight, self.bias, self.stride,
                self.padding)
        if not self.activation is None:
            output = self.activation(output)
            output, s_f = G_fix(output, bit = BIT_F)
            self.scale_F = s_f
        return output

class FixLinear(nn.Module):
    def __init__(self, input_features, output_features, bias=True, training=True):
        super(FixLinear, self).__init__()
        self.input_features = input_features
        self.output_features = output_features
        self.training = training

        # scale for S
        self.scale_F = 0
        self.scale_P_w = 0
        self.sclae_P_b = 0

        self.weight = nn.Parameter(torch.Tensor(output_features, input_features))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(output_features))
        else:
            self.register_parameter('bias', None)

        self.weight.data.uniform_(-0.1,0.1)
        if self.bias is not None:
            self.bias.data.uniform_(-0.1,0.1)


    def fix_parameter(self):
        weight_data = G_fix(self.weight, bit = BIT_P_w)
        self.weight.data = weight_data
        bias_data = G_fix(self.bias, bit = BIT_P_b)
        self.bias.data = bias_data

    def forward(self, input_data):
        self.fix_parameter()
        output = F.LinearFunction.apply(input_data, self.weight, self.bias)
        if not self.activation is None:
            output = self.activation(output)
            output, s_f = G_fix(output, bit = BIT_F)
            self.scale_F = s_f
        return output
