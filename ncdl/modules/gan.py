import torch
import torch.nn as nn
import numpy as np
from typing import Callable, List, Tuple, Optional


class MappingFC(nn.Module):
    def __init__(self,
                 in_features: int,
                 out_features: int,
                 activation: Callable = nn.Identity,
                 activation_gain: float = 1.0,
                 weight_gain: float = 1.0,
                 bias_gain: float = 0.0,
                 bias: bool = True):
        super(MappingFC, self).__init__()
        self.activation = activation()
        self.activation_gain = activation_gain
        self.weight = torch.nn.Parameter(torch.randn([out_features, in_features]) / weight_gain)
        self.bias = torch.nn.Parameter(torch.full([out_features], float(bias_gain))) if bias else None
        self.weight_gain = weight_gain / np.sqrt(in_features)
        self.bias_gain = weight_gain

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = self.weight * self.weight_gain
        b = self.bias
        if b and self.bias_gain != 1:
            b = b * self.bias_gain
        x = torch.addmm(b[None, ...], x, w.t())
        x = self.activation(x) * self.activation_gain
        return x


class MappingNetwork(torch.nn.Module):
    def __init__(self,
                 z_dim: int,
                 w_dim: int,
                 num_ws: Optional[int],
                 num_layers: int = 8,
                 activation: Callable = nn.LeakyReLU,
                 lr_multiplier=0.01,
                 w_avg_beta=0.995):
        super().__init__()
        self.z_dim = z_dim
        self.w_dim = w_dim
        self.num_ws = num_ws
        self.num_layers = num_layers
        self.w_avg_beta = w_avg_beta

        features_list = [z_dim] + [w_dim] * num_layers

        if num_ws is not None and w_avg_beta is not None:
            self.register_buffer('w_avg', torch.zeros([w_dim]))

        fcn = []
        for in_features, out_features in zip(features_list, features_list[1:]):
            layer = MappingFC(
                in_features,
                out_features,
                activation=activation,
                weight_gain=lr_multiplier)
            fcn += [layer]
        self.mapping = nn.Sequential(*fcn)

    def forward(self, z, psi=None, truncation_cutoff=None, skip_w_avg_update=False):
        z = z * (z.square().mean(dim=1, keepdim=True) + 1e-8).rsqrt()

        x = self.mapping(z)

        if self.w_avg_beta is not None and self.training and not skip_w_avg_update:
            self.w_avg.copy_(x.detach().mean(dim=0).lerp(self.w_avg, self.w_avg_beta))

        if self.num_ws is not None:
            x = x[:, None, :].repeat([1, self.num_ws, 1])

        # Apply truncation.
        if psi:
            if self.num_ws is None or truncation_cutoff is None:
                x = self.w_avg.lerp(x, psi)
            else:
                x[:, :truncation_cutoff] = self.w_avg.lerp(x[:, :truncation_cutoff], psi)

        return x
