#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 26 22:07:28 2022

@author: lds
"""
# import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
# from torch.nn import Parameter

# class FCBlock(nn.Module):
#     def __init__(self, in_dim, out_dim, use_act=True):
#         super().__init__()
#         self.use_act = use_act
#         self.fc = nn.Linear(in_dim, out_dim, bias=True)
#         self.act = nn.ReLU()
#         # self.drop = nn.Dropout(config.DROPRATE)

#     def forward(self, x):
#         return self.act(self.fc(x)) if self.use_act else self.fc(x)

# class ConvBlock(nn.Module):
#     def __init__(
#             self,
#             in_channels,
#             out_channels,
#             discriminator=False,
#             use_act=True,
#             use_bn=True,
#             **kwargs,
#     ):
#         super().__init__()
#         self.use_act = use_act
#         self.cnn = nn.Conv1d(in_channels, out_channels, **kwargs, bias=not use_bn)
#         self.bn = nn.BatchNorm1d(out_channels) if use_bn else nn.Identity()
#         self.act = (
#             nn.LeakyReLU(0.2, inplace=True)
#             if discriminator
#             else nn.PReLU(num_parameters=out_channels)
#         )
#         # self.drop = nn.Dropout(config.DROPRATE)

#     def forward(self, x):
#         return self.act(self.bn(self.cnn(x))) if self.use_act else self.bn(self.cnn(x))

# class ResidualBlock(nn.Module):
#     def __init__(self, in_channels):
#         super().__init__()
#         self.block1 = ConvBlock(
#             in_channels,
#             in_channels,
#             kernel_size=3,
#             stride=1,
#             padding=1
#         )
#         self.block2 = ConvBlock(
#             in_channels,
#             in_channels,
#             kernel_size=3,
#             stride=1,
#             padding=1,
#             use_act=False,
#         )

#     def forward(self, x):
#         out = self.block1(x)
#         out = self.block2(out)
#         return out + x

# def init_weights(net, init_type='normal', gain=0.02):
#     from torch.nn import init
#     def init_func(m):
#         classname = m.__class__.__name__
#         if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
#             if init_type == 'normal':
#                 init.normal(m.weight.data, 0.0, gain)
#             elif init_type == 'xavier':
#                 init.xavier_normal(m.weight.data, gain=gain)
#             elif init_type == 'kaiming':
#                 init.kaiming_normal(m.weight.data, a=0, mode='fan_in')
#             elif init_type == 'orthogonal':
#                 init.orthogonal(m.weight.data, gain=gain)
#             else:
#                 raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
#             if hasattr(m, 'bias') and m.bias is not None:
#                 init.constant(m.bias.data, 0.0)
#         elif classname.find('BatchNorm1d') != -1:
#             init.normal(m.weight.data, 1.0, gain)
#             init.constant(m.bias.data, 0.0)

#     print('initialize network with %s' % init_type)
#     net.apply(init_func)
    
# def get_pad(in_,  ksize, stride, atrous=1):
#     out_ = np.ceil(float(in_)/stride)
#     return int(((out_ - 1) * stride + atrous*(ksize-1) + 1 - in_)/2)

class GatedConv1dWithActivation(torch.nn.Module):
    """
    Gated Convlution layer with activation (default activation:LeakyReLU)
    Params: same as conv1d
    Input: The feature from last layer "I"
    Output:\phi(f(I))*\sigmoid(g(I))
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True,batch_norm=True, activation=torch.nn.LeakyReLU(0.2, inplace=True)):
        super(GatedConv1dWithActivation, self).__init__()
        self.batch_norm = batch_norm
        self.activation = activation
        self.conv1d = torch.nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)
        self.mask_conv1d = torch.nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)
        self.batch_norm1d = torch.nn.BatchNorm1d(out_channels)
        self.sigmoid = torch.nn.Sigmoid()

        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight)
                
    def gated(self, mask):
        #return torch.clamp(mask, -1, 1)
        return self.sigmoid(mask)
    
    def forward(self, input):
        raw = self.conv1d(input)
        mask = self.mask_conv1d(input)
        
        
        if self.activation is not None:
            x = self.activation(raw) * self.gated(mask)
        else:
            x = raw * self.gated(mask)
            
        if self.batch_norm:
            return self.batch_norm1d(x), raw, self.gated(mask)
        else:
            return x, raw, self.gated(mask)

class GatedDeConv1dWithActivation(torch.nn.Module):
    """
    Gated DeConvlution layer with activation (default activation:LeakyReLU)
    resize + conv
    Params: same as conv1d
    Input: The feature from last layer "I"
    Output:\phi(f(I))*\sigmoid(g(I))
    """
    def __init__(self, scale_factor, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, batch_norm=True,activation=torch.nn.LeakyReLU(0.2, inplace=True)):
        super(GatedDeConv1dWithActivation, self).__init__()
        self.gcovd1d = GatedConv1dWithActivation(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias, batch_norm, activation)
        self.scale_factor = scale_factor

    def forward(self, input):
        #print(input.size())
        x = F.interpolate(input, scale_factor=2)
        x, raw, score = self.gcovd1d(x)
        return x, raw, score

# class GatedConv2dWithActivation(torch.nn.Module):
#     """
#     Gated Convlution layer with activation (default activation:LeakyReLU)
#     Params: same as conv2d
#     Input: The feature from last layer "I"
#     Output:\phi(f(I))*\sigmoid(g(I))
#     """

#     def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True,batch_norm=True, activation=torch.nn.LeakyReLU(0.2, inplace=True)):
#         super(GatedConv2dWithActivation, self).__init__()
#         self.batch_norm = batch_norm
#         self.activation = activation
#         self.conv2d = torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)
#         self.mask_conv2d = torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)
#         self.batch_norm2d = torch.nn.BatchNorm2d(out_channels)
#         self.sigmoid = torch.nn.Sigmoid()

#         for m in self.modules():
#             if isinstance(m, nn.Conv2d):
#                 nn.init.kaiming_normal_(m.weight)
#     def gated(self, mask):
#         #return torch.clamp(mask, -1, 1)
#         return self.sigmoid(mask)
#     def forward(self, input):
#         x = self.conv2d(input)
#         mask = self.mask_conv2d(input)
#         if self.activation is not None:
#             x = self.activation(x) * self.gated(mask)
#         else:
#             x = x * self.gated(mask)
#         if self.batch_norm:
#             return self.batch_norm2d(x), self.gated(mask)
#         else:
#             return x, self.gated(mask)

# class GatedDeConv2dWithActivation(torch.nn.Module):
#     """
#     Gated DeConvlution layer with activation (default activation:LeakyReLU)
#     resize + conv
#     Params: same as conv2d
#     Input: The feature from last layer "I"
#     Output:\phi(f(I))*\sigmoid(g(I))
#     """
#     def __init__(self, scale_factor, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, batch_norm=True,activation=torch.nn.LeakyReLU(0.2, inplace=True)):
#         super(GatedDeConv2dWithActivation, self).__init__()
#         self.conv2d = GatedConv2dWithActivation(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias, batch_norm, activation)
#         self.scale_factor = scale_factor

#     def forward(self, input):
#         #print(input.size())
#         x = F.interpolate(input, scale_factor=2)
#         return self.conv2d(x)

# class SNGatedConv1dWithActivation(torch.nn.Module):
#     """
#     Gated Convolution with spetral normalization
#     """
#     def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, batch_norm=True, activation=torch.nn.LeakyReLU(0.2, inplace=True)):
#         super(SNGatedConv1dWithActivation, self).__init__()
#         self.conv1d = torch.nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)
#         self.mask_conv1d = torch.nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)
#         self.activation = activation
#         self.batch_norm = batch_norm
#         self.batch_norm1d = torch.nn.BatchNorm1d(out_channels)
#         self.sigmoid = torch.nn.Sigmoid()
#         # self.conv1d = torch.nn.utils.spectral_norm(self.conv1d)
#         # self.mask_conv1d = torch.nn.utils.spectral_norm(self.mask_conv1d)
#         for m in self.modules():
#             if isinstance(m, nn.Conv1d):
#                 nn.init.kaiming_normal_(m.weight)

#     def gated(self, mask):
#         return self.sigmoid(mask)
#         #return torch.clamp(mask, -1, 1)

#     def forward(self, input):
#         x = self.conv1d(input)
#         mask = self.mask_conv1d(input)
#         if self.activation is not None:
#             x = self.activation(x) * self.gated(mask)
#         else:
#             x = x * self.gated(mask)
#         if self.batch_norm:
#             return self.batch_norm1d(x)
#         else:
#             return x

# class SNGatedDeConv1dWithActivation(torch.nn.Module):
#     """
#     Gated DeConvlution layer with activation (default activation:LeakyReLU)
#     resize + conv
#     Params: same as conv1d
#     Input: The feature from last layer "I"
#     Output:\phi(f(I))*\sigmoid(g(I))
#     """
#     def __init__(self, scale_factor, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, batch_norm=True, activation=torch.nn.ReLU()):
#         super(SNGatedDeConv1dWithActivation, self).__init__()
#         self.conv1d = SNGatedConv1dWithActivation(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias, batch_norm, activation)
#         self.scale_factor = scale_factor

#     def forward(self, input):
#         #print(input.size())
#         x = F.interpolate(input, scale_factor=2)
#         return self.conv1d(x)

class SNConvWithActivation(torch.nn.Module):
    """
    SN convolution for spetral normalization conv
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, activation=torch.nn.LeakyReLU(0.2, inplace=True)):
        super(SNConvWithActivation, self).__init__()
        self.conv1d = torch.nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)
        # self.conv1d = torch.nn.utils.spectral_norm(self.conv1d)
        self.activation = activation
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight)
    def forward(self, input):
        x = self.conv1d(input)
        if self.activation is not None:
            return self.activation(x)
        else:
            return x

# def l2normalize(v, eps=1e-12):
#     return v / (v.norm() + eps)

# class SpectralNorm(nn.Module):
#     def __init__(self, module, name='weight', power_iterations=1):
#         super(SpectralNorm, self).__init__()
#         self.module = module
#         self.name = name
#         self.power_iterations = power_iterations
#         if not self._made_params():
#             self._make_params()

#     def _update_u_v(self):
#         u = getattr(self.module, self.name + "_u")
#         v = getattr(self.module, self.name + "_v")
#         w = getattr(self.module, self.name + "_bar")

#         height = w.data.shape[0]
#         for _ in range(self.power_iterations):
#             v.data = l2normalize(torch.mv(torch.t(w.view(height,-1).data), u.data))
#             u.data = l2normalize(torch.mv(w.view(height,-1).data, v.data))

#         # sigma = torch.dot(u.data, torch.mv(w.view(height,-1).data, v.data))
#         sigma = u.dot(w.view(height, -1).mv(v))
#         setattr(self.module, self.name, w / sigma.expand_as(w))

#     def _made_params(self):
#         try:
#             u = getattr(self.module, self.name + "_u")
#             v = getattr(self.module, self.name + "_v")
#             w = getattr(self.module, self.name + "_bar")
#             return True
#         except AttributeError:
#             return False


#     def _make_params(self):
#         w = getattr(self.module, self.name)

#         height = w.data.shape[0]
#         width = w.view(height, -1).data.shape[1]

#         u = Parameter(w.data.new(height).normal_(0, 1), requires_grad=False)
#         v = Parameter(w.data.new(width).normal_(0, 1), requires_grad=False)
#         u.data = l2normalize(u.data)
#         v.data = l2normalize(v.data)
#         w_bar = Parameter(w.data)

#         del self.module._parameters[self.name]

#         self.module.register_parameter(self.name + "_u", u)
#         self.module.register_parameter(self.name + "_v", v)
#         self.module.register_parameter(self.name + "_bar", w_bar)


#     def forward(self, *args):
#         self._update_u_v()
#         return self.module.forward(*args)

