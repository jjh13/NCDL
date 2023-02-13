import torch
import torch.nn as nn
from ncdl.lattice import LatticeTensor, Lattice
from torch.nn.modules.batchnorm import _NormBase
from ncdl.nn.functional.norm import batch_norm


class LatticeBatchNorm(_NormBase):
    def __init__(
        self,
        lattice: Lattice,
        num_features: int,
        eps: float = 1e-5,
        momentum: float = 0.1,
        affine: bool = True,
        track_running_stats: bool = True,
        device=None,
        dtype=None
    ) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super(LatticeBatchNorm, self).__init__(
            num_features, eps, momentum, affine, track_running_stats, **factory_kwargs
        )
        self._lattice = lattice

    def forward(self, input: LatticeTensor) -> LatticeTensor:
        if input.parent != self._lattice:
            raise ValueError("Input LatticeTensor belongs to the incorrect lattice.")
        # exponential_average_factor is set to self.momentum
        # (when it is available) only so that it gets updated
        # in ONNX graph when this node is exported to ONNX.
        if self.momentum is None:
            exponential_average_factor = 0.0
        else:
            exponential_average_factor = self.momentum

        if self.training and self.track_running_stats:
            # TODO: if statement only here to tell the jit to skip emitting this when it is None
            if self.num_batches_tracked is not None:  # type: ignore[has-type]
                self.num_batches_tracked.add_(1)  # type: ignore[has-type]
                if self.momentum is None:  # use cumulative moving average
                    exponential_average_factor = 1.0 / float(self.num_batches_tracked)
                else:  # use exponential moving average
                    exponential_average_factor = self.momentum

        r"""
        Decide whether the mini-batch stats should be used for normalization rather than the buffers.
        Mini-batch stats are used in training mode, and in eval mode when buffers are None.
        """
        if self.training:
            bn_training = True
        else:
            bn_training = (self.running_mean is None) and (self.running_var is None)

        r"""
        Buffers are only updated if they are to be tracked and we are in training mode. Thus they only need to be
        passed when the update should occur (i.e. in training mode when they are tracked), or when buffer stats are
        used for normalization (i.e. in eval mode when buffers are not None).
        """
        sliced = [None, slice(0, None)] + [None] * self._lattice.dimension
        return batch_norm(input,
                          self.running_mean[sliced] if not self.training or self.track_running_stats else None,
                          self.running_var[sliced] if not self.training or self.track_running_stats else None,
                          self.weight[sliced],
                          self.bias[sliced],
                          bn_training,
                          exponential_average_factor,
                          self.eps)