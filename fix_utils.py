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

def fix_weight(tensor, bit, is_first, fuse_tensor = None):
    tmp = tensor.detach()
    if fuse_tensor is None:
        tmp_min = torch.min(tmp) * RANGE_MIN if is_first else torch.min(tmp)
        tmp_max = torch.max(tmp) * RANGE_MAX if is_first else torch.max(tmp)
    else:
        tmp_fuse_tensor = fuse_tensor.detach()
        tmp_min = torch.min(tmp_fuse_tensor) * RANGE_MIN if is_first else torch.min(tmp_fuse_tensor)
        tmp_max = torch.max(tmp_fuse_tensor) * RANGE_MAX if is_first else torch.max(tmp_fuse_tensor)
    tmp = torch.clamp(tmp, min = tmp_min, max = tmp_max)
    scale = (tmp_max - tmp_min) / (2**bit - 1)
    zero = tmp_min
    tmp = (torch.round((tmp - tmp_min) /scale)) * scale + tmp_min
    return tmp, scale, zero

def fix_output(tensor, min_value, max_value, bit):
    tmp = tensor.detach()
    tmp = torch.clamp(tmp, min = min_value, max = max_value)
    scale = (max_value - min_value) / (2**bit - 1)
    zero = min_value
    tmp = (torch.round((tmp - min_value) /scale)) * scale + min_value
    return tmp, scale, zero

def fix_bias(tensor, s1, s2):
    tmp = tensor.detach()
    scale = s1 * s2 
    zero = 0
    tmp = torch.round(tmp / scale) * scale 
    return tmp, scale, zero

def get_scale(tensor, bit):
    tmp = tensor.detach()
    scale = (torch.max(tmp) - torch.min(tmp)) / (2**bit -1)
    return scale

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

        # fix parameters, just init
        self.scale_F = 0
        self.scale_P_w = 0
        self.scale_P_b = 0
        self.zero_F = -100
        self.zero_P_w = -100
        self.zero_P_b = -100

        # in the first time, need to deal with outleir
        self.is_first = True

        self.alpha = 0.9
        self.cur_max_F = 0
        self.cur_min_F = 0

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
        weight_temp, bias_temp, fuse_weight = None, None, None
        input_scale = get_scale(input, BIT_F)

        # use simulate the bn fusing 
        if self.bn is not None:
            assert self.bias is None
            fuse_weight = self.weight.clone().view(self.out_channels, -1)
            weight_bn = torch.diag(self.bn.weight.div(torch.sqrt(self.bn.eps + self.bn.running_var)))
            fuse_weight = torch.mm(weight_bn, fuse_weight)

        weight_copy = self.weight.data
        self.weight.data, self.scale_P_w, self.zero_P_w = fix_weight(self.weight, bit = BIT_P_w, is_first = self.is_first, fuse_tensor = fuse_weight)
        # weight_temp, self.scale_P_w, self.zero_P_w = fix_weight(self.weight, bit = BIT_P_w, is_first = self.is_first, fuse_tensor = fuse_weight)

        if self.bias is not None:
            assert self.bn is None
            bias_copy = self.bias.data
            self.bias.data, self.scale_P_b, self.zero_P_b = fix_bias(self.bias, input_scale, self.scale_P_w)
            # bias_temp, self.scale_P_b, self.zero_P_b = fix_bias(self.bias, input_scale, self.scale_P_w)

        output = F.conv2d(input, self.weight, self.bias, self.stride, self.padding)
        self.weight.data = weight_copy
        self.bias.data = bias_copy
        # output = F.conv2d(input, weight_temp, bias_temp, self.stride, self.padding)

        if self.bn is not None:
            bn_bias_copy = self.bn.bias.data
            self.bn.bias.data, self.scale_P_b, self.zero_P_b = fix_bias(self.bn.bias, input_scale, self.scale_P_w)
            # bn_bias_temp, self.scale_P_b, self.zero_P_b = fix_bias(self.bn.bias, input_scale, self.scale_P_w)

            output = self.bn(output)
            # output = F.batch_norm(output, self.bn.running_mean, self.bn.running_var, self.bn.weight, bn_bias_temp)
            self.bn.bias.data = bn_bias_copy

        if self.activation is not None:
            output = self.activation(output)

        with torch.no_grad():
            new_max = torch.max(output)
            new_min = torch.min(output)
            self.cur_max_F = self.cur_max_F + self.alpha * (new_max - self.cur_max_F)
            self.cur_min_F = self.cur_min_F + self.alpha * (new_min - self.cur_min_F)

        output.data, self.scale_F, self.zero_F = fix_output(output, self.cur_min_F.data, self.cur_max_F.data, bit = BIT_F)

        # after forward one time, just set is_first to False
        self.is_first = False
        return output
