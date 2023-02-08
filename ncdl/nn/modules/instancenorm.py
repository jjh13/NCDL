import torch
import torch.nn as nn
from ncdl.lattice import LatticeTensor, Lattice
from ncdl.nn.functional.convolution import lattice_conv
from ncdl.util.stencil import Stencil
from torch.nn.modules.batchnorm import _NormBase
from ncdl.nn.functional.norm import instance_norm


class LatticeInstanceNorm(_NormBase):
    def __init__(
        self,
        lattice: Lattice,
        num_features: int,
        eps: float = 1e-5,
        momentum: float = 0.1,
        affine: bool = False,
        track_running_stats: bool = False,
        device=None,
        dtype=None
    ) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super(LatticeInstanceNorm, self).__init__(
            num_features, eps, momentum, affine, track_running_stats, **factory_kwargs)
        self._lattice = lattice

    def _check_input_dim(self, input):
        raise NotImplementedError

    def _get_no_batch_dim(self):
        raise NotImplementedError

    def _apply_instance_norm(self, input):
        return instance_norm(input,
                             self.running_mean,
                             self.running_var,
                             self.weight,
                             self.bias,
                             self.training or not self.track_running_stats,
                             self.momentum,
                             self.eps)

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        version = local_metadata.get('version', None)
        # at version 1: removed running_mean and running_var when
        # track_running_stats=False (default)
        if version is None and not self.track_running_stats:
            running_stats_keys = []
            for name in ('running_mean', 'running_var'):
                key = prefix + name
                if key in state_dict:
                    running_stats_keys.append(key)
            if len(running_stats_keys) > 0:
                error_msgs.append(
                    'Unexpected running stats buffer(s) {names} for {klass} '
                    'with track_running_stats=False. If state_dict is a '
                    'checkpoint saved before 0.4.0, this may be expected '
                    'because {klass} does not track running stats by default '
                    'since 0.4.0. Please remove these keys from state_dict. If '
                    'the running stats are actually needed, instead set '
                    'track_running_stats=True in {klass} to enable them. See '
                    'the documentation of {klass} for details.'
                    .format(names=" and ".join('"{}"'.format(k) for k in running_stats_keys),
                            klass=self.__class__.__name__))
                for key in running_stats_keys:
                    state_dict.pop(key)

        super(LatticeInstanceNorm, self)._load_from_state_dict(
            state_dict, prefix, local_metadata, strict,
            missing_keys, unexpected_keys, error_msgs)

    def forward(self, input: LatticeTensor) -> LatticeTensor:
        if input.parent != self._lattice:
            raise ValueError("Input LatticeTensor belongs to the incorrect lattice.")

        return self._apply_instance_norm(input)