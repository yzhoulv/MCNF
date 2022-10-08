# Copyright (c) OpenMMLab. All rights reserved.
from functools import partial
from itertools import chain
from operator import truediv
from telnetlib import SE
from tkinter.font import names
from turtle import forward
from typing import Sequence
from timm.models.layers import trunc_normal_

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn.bricks import (NORM_LAYERS, DropPath, build_activation_layer,
                             build_norm_layer)
from mmcv.runner import BaseModule
from mmcv.runner.base_module import ModuleList, Sequential

from ..builder import BACKBONES
import numpy as np
import cv2


class BayarConv2D(nn.Module):
    def __init__(self, in_channels=3, out_channels=3) -> None:
        super().__init__()
        self.BayarConv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, stride=1, kernel_size=5, padding=2)
        self.bayar_mask = (torch.tensor(np.ones(shape=(5, 5)))).cuda()
        self.bayar_mask[2, 2] = 0
        self.bayar_final = (torch.tensor(np.zeros((5, 5)))).cuda()
        self.bayar_final[2, 2] = -1
        self.norm = nn.BatchNorm2d(3)
    
    def forward(self, input):
        self.BayarConv.weight.data *= self.bayar_mask
        self.BayarConv.weight.data *= torch.pow(self.BayarConv.weight.data.sum(axis=(2, 3)).view(3, 3, 1, 1), -1)
        self.BayarConv.weight.data += self.bayar_final
        x = self.BayarConv(input)
        return self.norm(x)

class SRMConv2D(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, stride=1, padding=2):
        super(SRMConv2D, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride
        self.padding = padding
        self.SRMWeights = nn.Parameter(
            self._get_srm_list(), requires_grad=False)
        self.norm = nn.BatchNorm2d(3)

    def _get_srm_list(self):
        # srm kernel 1
        srm1 = [[0,  0, 0,  0, 0],
                [0, -1, 2, -1, 0],
                [0,  2, -4, 2, 0],
                [0, -1, 2, -1, 0],
                [0,  0, 0,  0, 0]]
        srm1 = torch.tensor(srm1, dtype=torch.float32) / 4.

        # srm kernel 2
        srm2 = [[-1, 2, -2, 2, -1],
                [2, -6, 8, -6, 2],
                [-2, 8, -12, 8, -2],
                [2, -6, 8, -6, 2],
                [-1, 2, -2, 2, -1]]
        srm2 = torch.tensor(srm2, dtype=torch.float32) / 12.

        # srm kernel 3
        srm3 = [[0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0],
                [0, 1, -2, 1, 0],
                [0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0]]
        srm3 = torch.tensor(srm3, dtype=torch.float32) / 2.

        return torch.stack([torch.stack([srm1, srm1, srm1], dim=0), torch.stack([srm2, srm2, srm2], dim=0), torch.stack([srm3, srm3, srm3], dim=0)], dim=0)

    def forward(self, input):
        # X1 =
        x = F.conv2d(input, self.SRMWeights, stride=self.stride, padding=self.padding)
        return self.norm(x)

class SobelConv2D(nn.Module):
    def __init__(self, in_channels=3, out_channels=3) -> None:
        super(SobelConv2D, self).__init__()
        filter_x = np.array([
        [1, 0, -1],
        [2, 0, -2],
        [1, 0, -1],
        ]).astype(np.float32)
        filter_y = np.array([
            [1, 2, 1],
            [0, 0, 0],
            [-1, -2, -1],
        ]).astype(np.float32)

        filter_x = filter_x.reshape((1, 1, 3, 3))
        filter_x = np.repeat(filter_x, in_channels, axis=1)
        filter_x = np.repeat(filter_x, out_channels, axis=0)

        filter_y = filter_y.reshape((1, 1, 3, 3))
        filter_y = np.repeat(filter_y, in_channels, axis=1)
        filter_y = np.repeat(filter_y, out_channels, axis=0)

        filter_x = torch.from_numpy(filter_x)
        filter_y = torch.from_numpy(filter_y)
        filter_x = nn.Parameter(filter_x, requires_grad=False)
        filter_y = nn.Parameter(filter_y, requires_grad=False)
        conv_x = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        conv_x.weight = filter_x
        conv_y = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        conv_y.weight = filter_y
        self.sobel_x = nn.Sequential(conv_x, nn.BatchNorm2d(out_channels))
        self.sobel_y = nn.Sequential(conv_y, nn.BatchNorm2d(out_channels))
    
    def forward(self, input):
        g_x = self.sobel_x(input)
        g_y = self.sobel_y(input)
        g = torch.sqrt(torch.pow(g_x, 2) + torch.pow(g_y, 2))
        return torch.sigmoid(g) * input

class HighPassConv2D(nn.Module):
    def __init__(self, in_channels=3, out_channels=3) -> None:
        super(HighPassConv2D, self).__init__()
        self.depthwise_conv = nn.Conv2d(
            in_channels,
            in_channels,
            kernel_size=7,
            padding=3,
            groups=in_channels)

        self.hp_conv_d1 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, stride=1, kernel_size=3, padding='same', groups=in_channels)
        self.hp_conv_d2 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, stride=1, kernel_size=3, padding='same', groups=in_channels)
        self.hp_conv_d3 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, stride=1, kernel_size=5, padding='same', groups=in_channels)
        self.hp_conv_d4 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, stride=1, kernel_size=5, padding='same', groups=in_channels)

        self.filters = {
            'd1': [
                np.array([[0., 0., 0.], [0., -1., 0.], [0., 1., 0.]]),
                np.array([[0., 0., 0.], [0., -1., 1.], [0., 0., 0.]]),
                np.array([[0., 0., 0.], [0., -1., 0.], [0., 0., 1.]])],
            'd2': [
                np.array([[0., 1., 0.], [0., -2., 0.], [0., 1., 0.]]),
                np.array([[0., 0., 0.], [1., -2., 1.], [0., 0., 0.]]),
                np.array([[1., 0., 0.], [0., -2., 0.], [0., 0., 1.]])],
            'd3': [
                np.array([[0., 0., 0., 0., 0.], [0., 0., -1., 0., 0.], [0., 0., 3., 0., 0.], [0., 0., -3., 0., 0.], [0., 0., 1., 0., 0.]]),
                np.array([[0., 0., 0., 0., 0.], [0., 0., 0., 0., 0.], [0., -1., 3., -3., 1.], [0., 0., 0., 0., 0.], [0., 0., 0., 0., 0.]]),
                np.array([[0., 0., 0., 0., 0.], [0., -1., 0., 0., 0.], [0., 0., 3., 0., 0.], [0., 0., 0., -3., 0.], [0., 0., 0., 0., 1.]])],
            'd4': [
                np.array([[0., 0., 1., 0., 0.], [0., 0., -4., 0., 0.], [0., 0., 6., 0., 0.], [0., 0., -4., 0., 0.], [0., 0., 1., 0., 0.]]),
                np.array([[0., 0., 0., 0., 0.], [0., 0., 0., 0., 0.], [1., -4., 6., -4., 1.], [0., 0., 0., 0., 0.], [0., 0., 0., 0., 0.]]),
                np.array([[1., 0., 0., 0., 0.], [0., -4., 0., 0., 0.], [0., 0., 6., 0., 0.], [0., 0., 0., -4., 0.], [0., 0., 0., 0., 1.]])],
        }
        self.hp_conv_d1.weight.data = torch.tensor(self.filters['d1'], dtype=torch.float, requires_grad=True).unsqueeze(1).cuda()
        self.hp_conv_d2.weight.data = torch.tensor(self.filters['d2'], dtype=torch.float, requires_grad=True).unsqueeze(1).cuda()
        self.hp_conv_d3.weight.data = torch.tensor(self.filters['d3'], dtype=torch.float, requires_grad=True).unsqueeze(1).cuda()
        self.hp_conv_d4.weight.data = torch.tensor(self.filters['d4'], dtype=torch.float, requires_grad=True).unsqueeze(1).cuda()
        
        self.conv = nn.Conv2d(in_channels=in_channels*4, out_channels=3, kernel_size=1, stride=1, padding=0, bias=False)
        self.norm = nn.BatchNorm2d(3)
            
    def forward(self, input):
        d1 = self.hp_conv_d1(input)
        d2 = self.hp_conv_d2(input)
        d3 = self.hp_conv_d3(input)
        d4 = self.hp_conv_d4(input)
        x = torch.cat([d1, d2, d3, d4], dim=1)
        x = self.conv(x)
        return self.norm(x)

class DCTConv2D(nn.Module): 
    def __init__(self) -> None:
        super(DCTConv2D, self).__init__()
        self.dc_layer0_dil = nn.Sequential(
            nn.Conv2d(in_channels=21,
                      out_channels=64,
                      kernel_size=3,
                      stride=1,
                      dilation=8,
                      padding=8),
            nn.BatchNorm2d(64, momentum=0.01),
            nn.ReLU(inplace=True)
        )
        self.dc_layer1_tail = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=4, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(4, momentum=0.01),
            nn.ReLU(inplace=True)
        )
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=3, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(3)
        )


    def convert_dct_coe(self, t_DCT_coefs):
        t_DCT_vols = []
        for i in range(t_DCT_coefs.shape[0]):
            t_DCT_coef = t_DCT_coefs[i]
            T = 20
            t_DCT_vol = torch.zeros(size=(T+1, t_DCT_coef.shape[1], t_DCT_coef.shape[2])).cuda()
            t_DCT_vol[0] += (t_DCT_coef == 0).float().squeeze()
            for i in range(1, T):
                t_DCT_vol[i] += (t_DCT_coef == i).float().squeeze()
                t_DCT_vol[i] += (t_DCT_coef == -i).float().squeeze()
            t_DCT_vol[T] += (t_DCT_coef >= T).float().squeeze()
            t_DCT_vol[T] += (t_DCT_coef <= -T).float().squeeze()
            t_DCT_vols.append(t_DCT_vol)

        return torch.stack(t_DCT_vols, dim=0)

    
    def forward(self, input, qtable):
        qtable = qtable.cuda()
        DCTcoef = self.convert_dct_coe(input)
        x = self.dc_layer0_dil(DCTcoef)
        x = self.dc_layer1_tail(x)
        B, C, H, W = x.shape
        x0 = x.reshape(B, C, H // 8, 8, W // 8, 8).permute(0, 1, 3, 5, 2, 4).reshape(B, 64 * C, H // 8,
                                                                                     W // 8)  # [B, 256, 32, 32]
        x_temp = x.reshape(B, C, H // 8, 8, W // 8, 8).permute(0, 1, 3, 5, 2, 4)  # [B, C, 8, 8, 32, 32]
        q_temp = qtable.unsqueeze(-1).unsqueeze(-1).unsqueeze(1)  # [B, 1, 8, 8, 1, 1]
        xq_temp = x_temp * q_temp  # [B, C, 8, 8, 32, 32]
        x1 = xq_temp.reshape(B, 64 * C, H // 8, W // 8)  # [B, 256, 32, 32]
        x = torch.cat([x0, x1], dim=1)
        x = self.conv(x)
        x = nn.functional.interpolate(x, scale_factor=8, mode='bilinear', align_corners=False) 
        return x

# @NORM_LAYERS.register_module('LN2d')
class LayerNorm2d(nn.LayerNorm):
    """LayerNorm on channels for 2d images.

    Args:
        num_channels (int): The number of channels of the input tensor.
        eps (float): a value added to the denominator for numerical stability.
            Defaults to 1e-5.
        elementwise_affine (bool): a boolean value that when set to ``True``,
            this module has learnable per-element affine parameters initialized
            to ones (for weights) and zeros (for biases). Defaults to True.
    """

    def __init__(self, num_channels: int, **kwargs) -> None:
        super().__init__(num_channels, **kwargs)
        self.num_channels = self.normalized_shape[0]

    def forward(self, x):
        assert x.dim() == 4, 'LayerNorm2d only supports inputs with shape ' \
            f'(N, C, H, W), but got tensor with shape {x.shape}'
        return F.layer_norm(
            x.permute(0, 2, 3, 1), self.normalized_shape, self.weight,
            self.bias, self.eps).permute(0, 3, 1, 2)

class ConvNeXtBlock(BaseModule):
    """ConvNeXt Block.

    Args:
        in_channels (int): The number of input channels.
        norm_cfg (dict): The config dict for norm layers.
            Defaults to ``dict(type='LN2d', eps=1e-6)``.
        act_cfg (dict): The config dict for activation between pointwise
            convolution. Defaults to ``dict(type='GELU')``.
        mlp_ratio (float): The expansion ratio in both pointwise convolution.
            Defaults to 4.
        linear_pw_conv (bool): Whether to use linear layer to do pointwise
            convolution. More details can be found in the note.
            Defaults to True.
        drop_path_rate (float): Stochastic depth rate. Defaults to 0.
        layer_scale_init_value (float): Init value for Layer Scale.
            Defaults to 1e-6.

    Note:
        There are two equivalent implementations:

        1. DwConv -> LayerNorm -> 1x1 Conv -> GELU -> 1x1 Conv;
           all outputs are in (N, C, H, W).
        2. DwConv -> LayerNorm -> Permute to (N, H, W, C) -> Linear -> GELU
           -> Linear; Permute back

        As default, we use the second to align with the official repository.
        And it may be slightly faster.
    """

    def __init__(self,
                 in_channels,
                 norm_cfg=dict(type='LN2d', eps=1e-6),
                 act_cfg=dict(type='GELU'),
                 mlp_ratio=4.,
                 linear_pw_conv=True,
                 drop_path_rate=0.,
                 layer_scale_init_value=1e-6):
        super().__init__()
        self.depthwise_conv = nn.Conv2d(
            in_channels,
            in_channels,
            kernel_size=7,
            padding=3,
            groups=in_channels)

        self.linear_pw_conv = linear_pw_conv
        self.norm = build_norm_layer(norm_cfg, in_channels)[1]

        mid_channels = int(mlp_ratio * in_channels)
        if self.linear_pw_conv:
            # Use linear layer to do pointwise conv.
            pw_conv = nn.Linear
        else:
            pw_conv = partial(nn.Conv2d, kernel_size=1)

        self.pointwise_conv1 = pw_conv(in_channels, mid_channels)
        self.act = build_activation_layer(act_cfg)
        self.pointwise_conv2 = pw_conv(mid_channels, in_channels)

        self.gamma = nn.Parameter(
            layer_scale_init_value * torch.ones((in_channels)),
            requires_grad=True) if layer_scale_init_value > 0 else None

        self.drop_path = DropPath(
            drop_path_rate) if drop_path_rate > 0. else nn.Identity()

    def forward(self, x):
        shortcut = x
        x = self.depthwise_conv(x)
        x = self.norm(x)

        if self.linear_pw_conv:
            x = x.permute(0, 2, 3, 1)  # (N, C, H, W) -> (N, H, W, C)

        x = self.pointwise_conv1(x)
        x = self.act(x)
        x = self.pointwise_conv2(x)

        if self.linear_pw_conv:
            x = x.permute(0, 3, 1, 2)  # permute back

        if self.gamma is not None:
            x = x.mul(self.gamma.view(1, -1, 1, 1))

        x = shortcut + self.drop_path(x)
        return x

class SELayer(nn.Module):
    def __init__(self, channel, reduction=2):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)# Squeeze操作的定义
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(# Excitation操作的定义
            nn.Linear(channel, channel * reduction, bias=False),# 压缩
            nn.ReLU(inplace=True),
            nn.Linear(channel * reduction, channel, bias=False),# 恢复
            nn.Sigmoid()# 定义归一化操作
        )

    def _save_weight(self, t):
        f = open('/media/scu/scu/datasets/Manipulation/NIST/weight.txt', 'a')
        np.set_printoptions(suppress=True)
        t_np = t.cpu().detach().numpy()
        t_np = np.squeeze(t_np)
        t_np = t_np.reshape(5, 3)
        t_np = np.sum(t_np, axis=1) / 3.0
        t_np = 1 / (1 + np.exp(-t_np))
        f.write(str(t_np) + '\n')
        f.close()
        


    def forward(self, x):
        b, c, _, _ = x.size()# 得到H和W的维度，在这两个维度上进行全局池化
        y_avg = self.avg_pool(x).view(b, c)# Squeeze操作的实现
        y_max = self.max_pool(x).view(b, c)
        y = y_avg + y_max
        y = self.fc(y).view(b, c, 1, 1)# Excitation操作的实现
        self._save_weight(y)
        # 将y扩展到x相同大小的维度后进行赋权
        return x * y.expand_as(x)

class ConvNeXt(BaseModule):
    """ConvNeXt.

    A PyTorch implementation of : `A ConvNet for the 2020s
    <https://arxiv.org/pdf/2201.03545.pdf>`_

    Modified from the `official repo
    <https://github.com/facebookresearch/ConvNeXt/blob/main/models/convnext.py>`_
    and `timm
    <https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/convnext.py>`_.

    Args:
        arch (str | dict): The model's architecture. If string, it should be
            one of architecture in ``ConvNeXt.arch_settings``. And if dict, it
            should include the following two keys:

            - depths (list[int]): Number of blocks at each stage.
            - channels (list[int]): The number of channels at each stage.

            Defaults to 'tiny'.
        in_channels (int): Number of input image channels. Defaults to 3.
        stem_patch_size (int): The size of one patch in the stem layer.
            Defaults to 4.
        norm_cfg (dict): The config dict for norm layers.
            Defaults to ``dict(type='LN2d', eps=1e-6)``.
        act_cfg (dict): The config dict for activation between pointwise
            convolution. Defaults to ``dict(type='GELU')``.
        linear_pw_conv (bool): Whether to use linear layer to do pointwise
            convolution. Defaults to True.
        drop_path_rate (float): Stochastic depth rate. Defaults to 0.
        layer_scale_init_value (float): Init value for Layer Scale.
            Defaults to 1e-6.
        out_indices (Sequence | int): Output from which stages.
            Defaults to -1, means the last stage.
        frozen_stages (int): Stages to be frozen (all param fixed).
            Defaults to 0, which means not freezing any parameters.
        gap_before_final_norm (bool): Whether to globally average the feature
            map before the final norm layer. In the official repo, it's only
            used in classification task. Defaults to True.
        init_cfg (dict, optional): Initialization config dict
    """  # noqa: E501
    arch_settings = {
        'tiny': {
            'depths': [3, 3, 9, 3],
            'channels': [96, 192, 384, 768]
        },
        'small': {
            'depths': [3, 3, 27, 3],
            'channels': [96, 192, 384, 768]
        },
        'base': {
            'depths': [3, 3, 27, 3],
            'channels': [128, 256, 512, 1024]
        },
        'large': {
            'depths': [3, 3, 27, 3],
            'channels': [192, 384, 768, 1536]
        },
        'xlarge': {
            'depths': [3, 3, 27, 3],
            'channels': [256, 512, 1024, 2048]
        },
    }

    def __init__(self,
                 arch='small',
                 in_channels=12,
                 stem_patch_size=4,
                 norm_cfg=dict(type='LN2d', eps=1e-6),
                 act_cfg=dict(type='GELU'),
                 linear_pw_conv=True,
                 drop_path_rate=0.4,
                 layer_scale_init_value=1.0,
                 out_indices=[0, 1, 2, 3],
                 frozen_stages=0,
                 gap_before_final_norm=False,
                 init_cfg=None):
        super().__init__(init_cfg=init_cfg)

        if isinstance(arch, str):
            assert arch in self.arch_settings, \
                f'Unavailable arch, please choose from ' \
                f'({set(self.arch_settings)}) or pass a dict.'
            arch = self.arch_settings[arch]
        elif isinstance(arch, dict):
            assert 'depths' in arch and 'channels' in arch, \
                f'The arch dict must have "depths" and "channels", ' \
                f'but got {list(arch.keys())}.'

        self.depths = arch['depths']
        self.channels = arch['channels']
        assert (isinstance(self.depths, Sequence)
                and isinstance(self.channels, Sequence)
                and len(self.depths) == len(self.channels)), \
            f'The "depths" ({self.depths}) and "channels" ({self.channels}) ' \
            'should be both sequence with the same length.'

        self.num_stages = len(self.depths)

        if isinstance(out_indices, int):
            out_indices = [out_indices]
        assert isinstance(out_indices, Sequence), \
            f'"out_indices" must by a sequence or int, ' \
            f'get {type(out_indices)} instead.'
        for i, index in enumerate(out_indices):
            if index < 0:
                out_indices[i] = 4 + index
                assert out_indices[i] >= 0, f'Invalid out_indices {index}'
        self.out_indices = out_indices

        self.frozen_stages = frozen_stages
        self.gap_before_final_norm = gap_before_final_norm

        # stochastic depth decay rule
        dpr = [
            x.item()
            for x in torch.linspace(0, drop_path_rate, sum(self.depths))
        ]
        block_idx = 0

        # 4 downsample layers between stages, including the stem layer.
        self.downsample_layers = ModuleList()
        stem = nn.Sequential(
            nn.Conv2d(
                in_channels,
                self.channels[0],
                kernel_size=stem_patch_size,
                stride=stem_patch_size),
            build_norm_layer(norm_cfg, self.channels[0])[1],
        )
        self.downsample_layers.append(stem)

        # 4 feature resolution stages, each consisting of multiple residual
        # blocks
        self.stages = nn.ModuleList()

        for i in range(self.num_stages):
            depth = self.depths[i]
            channels = self.channels[i]

            if i >= 1:
                downsample_layer = nn.Sequential(
                    LayerNorm2d(self.channels[i - 1]),
                    nn.Conv2d(
                        self.channels[i - 1],
                        channels,
                        kernel_size=2,
                        stride=2),
                )
                self.downsample_layers.append(downsample_layer)

            stage = Sequential(*[
                ConvNeXtBlock(
                    in_channels=channels,
                    drop_path_rate=dpr[block_idx + j],
                    norm_cfg=norm_cfg,
                    act_cfg=act_cfg,
                    linear_pw_conv=linear_pw_conv,
                    layer_scale_init_value=layer_scale_init_value)
                for j in range(depth)
            ])
            block_idx += depth

            self.stages.append(stage)

            if i in self.out_indices:
                norm_layer = build_norm_layer(norm_cfg, channels)[1]
                self.add_module(f'norm{i}', norm_layer)

        self._freeze_stages()


    def forward(self, x):
        outs = []
        for i, stage in enumerate(self.stages):
            x = self.downsample_layers[i](x)
            x = stage(x)
            if i in self.out_indices:
                norm_layer = getattr(self, f'norm{i}')
                if self.gap_before_final_norm:
                    gap = x.mean([-2, -1], keepdim=True)
                    outs.append(norm_layer(gap).flatten(1))
                else:
                    # The output of LayerNorm2d may be discontiguous, which
                    # may cause some problem in the downstream tasks
                    outs.append(norm_layer(x).contiguous())

        return tuple(outs)

    def _freeze_stages(self):
        for i in range(self.frozen_stages):
            downsample_layer = self.downsample_layers[i]
            stage = self.stages[i]
            downsample_layer.eval()
            stage.eval()
            for param in chain(downsample_layer.parameters(),
                               stage.parameters()):
                param.requires_grad = False

    def train(self, mode=True):
        super(ConvNeXt, self).train(mode)
        self._freeze_stages()


@BACKBONES.register_module()
class MyNet(BaseModule):
    def __init__(self, init_cfg):
        super().__init__()
        # self.checkpoint_file = 'https://download.openmmlab.com/mmclassification/v0/convnext/downstream/convnext-tiny_3rdparty_32xb128-noema_in1k_20220301-795e9634.pth'
        # self.checkpoint_file = 'https://download.openmmlab.com/mmclassification/v0/convnext/downstream/convnext-base_3rdparty_32xb128-noema_in1k_20220301-2a0ee547.pth'  # noqa'
        self.checkpoint_file = 'https://download.openmmlab.com/mmclassification/v0/convnext/downstream/convnext-small_3rdparty_32xb128-noema_in1k_20220301-303e75e3.pth'
        self.init_cfg_noise=dict(
            type='Pretrained', 
            checkpoint = self.checkpoint_file,
            prefix='backbone.')
        
        self.bayerConv = BayarConv2D().cuda()
        self.SRMConv = SRMConv2D().cuda()
        self.sobelConv = SobelConv2D().cuda()
        self.highPassConv = HighPassConv2D().cuda()
        self.rgbConv = nn.Conv2d(in_channels=3, out_channels=3, kernel_size=5, stride=1, padding='same', bias=False)
        self.dctConv = DCTConv2D().cuda()
        self.dwConv = nn.Sequential(
            # nn.BatchNorm2d(15),
            nn.Conv2d(in_channels=15, out_channels=15, kernel_size=3, stride=1, padding='same', groups=15),
            # nn.Conv2d(in_channels=15, out_channels=15, kernel_size=1, stride=1, padding='same', groups=1),
            LayerNorm2d(15),
            # nn.ReLU(inplace=True)
        )
        self.atten = SELayer(channel=15)
        self.rgbStream = ConvNeXt(in_channels=3, init_cfg=init_cfg)
        self.noiseStream = ConvNeXt(in_channels=15, init_cfg=self.init_cfg)
    
    def _save_info_img(self, t, name):
        data_np = t.cpu().detach().numpy()
        data_np = np.squeeze(data_np)
        data_norm = (data_np-data_np.min()) / (data_np.max() - data_np.min()) * 255
        data_norm = data_norm.swapaxes(0, 2).swapaxes(0, 1)
        # cv2.imwrite('/media/scu/scu/datasets/Manipulation/images/res/' + name + '.png', data_norm.astype(np.uint8))
        cv2.imwrite('/media/scu/scu/datasets/Manipulation/images/res/' + name + '.png', \
                cv2.cvtColor(data_norm.astype(np.uint8), cv2.COLOR_BGR2RGB))
    
    def _save_depthwise_info(self, t, d):
        data_np = t.cpu().detach().numpy()
        data_np = np.squeeze(data_np)
        if d == True:
            names = ['bayer_dwc', 'srm_dwc', 'highPass_dwc', 'sobel_dwc', 'dct_dwc']
        else:
            names = ['bayer_w', 'srm_w', 'highPass_w', 'sobel_w', 'dct_w']
        for i in range(0, 15, 3):
            data = data_np[i:i+3, :, :]
            data_norm = (data-data.min()) / (data.max() - data.min()) * 255
            data_norm = data_norm.swapaxes(0, 2).swapaxes(0, 1)
            cv2.imwrite('/media/scu/scu/datasets/Manipulation/images/res/' + names[i//3] + '.png', \
                cv2.cvtColor(data_norm.astype(np.uint8), cv2.COLOR_BGR2RGB))
            # cv2.imwrite('/media/scu/scu/datasets/Manipulation/images/res/' + names[i//3] + '.png', data_norm.astype(np.uint8))



    def forward(self, x, qtabels):
        x_rgb = x[:, 0:3, :, :]
        x_dct = x[:, 3:, :, :]
        bayer = self.bayerConv(x_rgb)
        srm = self.SRMConv(x_rgb)
        sobel = self.sobelConv(x_rgb)
        highPass = self.highPassConv(x_rgb)
        dct = self.dctConv(x_dct, qtabels)
        # self._save_info_img(bayer, 'bayer')
        # self._save_info_img(srm, 'srm')
        # self._save_info_img(sobel, 'sobel')
        # self._save_info_img(highPass, 'highPass')
        # self._save_info_img(dct, 'dct')
        noise = torch.cat([bayer, srm, highPass, sobel, dct], dim=1)
        noise = self.dwConv(noise)
        # self._save_depthwise_info(noise, d=True)
        noise = self.atten(noise)
        # self._save_depthwise_info(noise, d=False)
        out_rgb = self.rgbStream(x_rgb)
        out_noise = self.noiseStream(noise)
        return tuple([out_rgb[2], out_rgb[3]]), tuple([out_noise[2], out_noise[3]])
        # return tuple([out_rgb[0], out_rgb[1], torch.add(out_noise[2], out_rgb[2]), torch.add(out_rgb[3], out_noise[3])])