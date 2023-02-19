from ncdl.lattice import Lattice, LatticeTensor
from math import prod
from typing import Optional
from torch import Tensor
import torch


def coset_moments(lt: LatticeTensor, groups=None, dims=None):
    batches, channels, *idims = lt.coset(0).shape
    numel = 0

    if dims is None:
        dims = [-(_+1) for _ in range(lt.parent.dimension)]

    if groups is not None:
        assert 1 not in dims
        assert -(lt.parent.dimension + 2) not in dims
        assert channels % groups == 0
        dims = [_ + 1 if _ >= 2 else _ for _ in dims]

        # Make sure the sums happen over the grouped channels
        dims += [2]

    sums, sumsq, grouped_cosets = [], [], []
    for coset_index in  range(lt.parent.coset_count):
        if groups is  None:
            shape = lt.coset(coset_index).shape
            numel += prod([shape[_] for _ in dims])
            sums += [lt.coset(coset_index).sum(dims, keepdim=True)]
            sumsq += [lt.coset(coset_index).pow(2).sum(dims, keepdim=True)]
        else:
            coset_ref = lt.coset(coset_index)
            grouped = coset_ref.reshape(*(batches, groups, channels//groups, *idims))
            numel += prod([grouped.shape[_] for _ in dims])
            grouped_cosets += [grouped]

            s = grouped.sum(dims, keepdim=True)
            sq = grouped.pow(2).sum(dims, keepdim=True)

            sums += [s]
            sumsq += [sq]

    sum_ = sum(sums)
    mean = sum_ / numel
    sumvar = sum(sumsq) - sum_ * mean
    return grouped_cosets, mean, sumvar, numel


def base_norm(
        lt: LatticeTensor,
        running_mean: Optional[Tensor] = None,
        running_var: Optional[Tensor] = None,
        weight: Optional[Tensor] = None,
        bias: Optional[Tensor] = None,
        use_input_stats: bool = True,
        momentum: float = 0.1,
        eps: float = 1e-5,
        dims=None,
        groups=None
    ) -> LatticeTensor:
    batches, channels, *idims = lt.coset(0).shape

    grouped_cosets = None

    if use_input_stats:
        grouped_cosets, mean, var_sum, numels = coset_moments(lt, dims=dims, groups=groups)

        if running_mean is not None:
            with torch.no_grad():
                mean_ = torch.resize_as_(mean.mean(dim=[0, 2, 3], keepdim=True), running_mean)
                running_mean.copy_((1 - momentum) * running_mean + momentum * mean_)
        if running_var is not None:
            with torch.no_grad():
                var_ = torch.resize_as_((var_sum/(numels - 1)).mean(dim=[0, 2, 3], keepdim=True), running_var)
                running_var.copy_((1 - momentum) * running_var + momentum * var_)
        var = var_sum/numels
    else:
        slice_ = [None, slice(0, None)] + [None] * lt.parent.dimension
        mean, var = running_mean[slice_], running_var[slice_]

    std = (var + eps).sqrt()

    cosets = []
    for coset_index in range(lt.parent.coset_count):
        if groups is None:
            coset = (lt.coset(coset_index) - mean)/std
        else:
            coset = (grouped_cosets[coset_index] - mean)/std
            coset = coset.reshape(*(batches, channels, *idims))

        if weight is not None:
            coset *= weight

        if bias is not None:
            coset += bias
        cosets += [coset]

    return LatticeTensor(None, parent=lt.parent, alt_cosets=cosets, alt_offsets=lt.coset_vectors)


def instance_norm(
        lt: LatticeTensor,
        running_mean: Optional[Tensor] = None,
        running_var: Optional[Tensor] = None,
        weight: Optional[Tensor] = None,
        bias: Optional[Tensor] = None,
        use_input_stats: bool = True,
        momentum: float = 0.1,
        eps: float = 1e-5,
    ) -> LatticeTensor:
    dims = [-(1 + _) for _ in range(lt.parent.dimension)]
    return base_norm(lt, running_mean, running_var, weight, bias, use_input_stats, momentum, eps, dims)


def layer_norm(
        lt: LatticeTensor,
        running_mean: Optional[Tensor] = None,
        running_var: Optional[Tensor] = None,
        weight: Optional[Tensor] = None,
        bias: Optional[Tensor] = None,
        use_input_stats: bool = True,
        momentum: float = 0.1,
        eps: float = 1e-5,
    ) -> LatticeTensor:
    dims = [1] + [-(1 + _) for _ in range(lt.parent.dimension)]
    return base_norm(lt, running_mean, running_var, weight, bias, use_input_stats, momentum, eps, dims)


def group_norm(
        lt: LatticeTensor,
        groups: int,
        running_mean: Optional[Tensor] = None,
        running_var: Optional[Tensor] = None,
        weight: Optional[Tensor] = None,
        bias: Optional[Tensor] = None,
        use_input_stats: bool = True,
        momentum: float = 0.1,
        eps: float = 1e-5,
    ) -> LatticeTensor:
    dims = [-(1 + _) for _ in range(lt.parent.dimension)]
    return base_norm(lt, running_mean, running_var, weight, bias, use_input_stats, momentum, eps, dims, groups=groups)


def gbatch_norm(
        lt: LatticeTensor,
        groups: int,
        running_mean: Optional[Tensor] = None,
        running_var: Optional[Tensor] = None,
        weight: Optional[Tensor] = None,
        bias: Optional[Tensor] = None,
        use_input_stats: bool = True,
        momentum: float = 0.1,
        eps: float = 1e-5,
    ) -> LatticeTensor:
    dims = [0] + [-(1 + _) for _ in range(lt.parent.dimension)]
    return base_norm(lt, running_mean, running_var, weight, bias, use_input_stats, momentum, eps, dims, groups=groups)


def batch_norm(
        lt: LatticeTensor,
        running_mean: Optional[Tensor] = None,
        running_var: Optional[Tensor] = None,
        weight: Optional[Tensor] = None,
        bias: Optional[Tensor] = None,
        use_input_stats: bool = True,
        momentum: float = 0.1,
        eps: float = 1e-5,
    ) -> LatticeTensor:
    dims = [0] + [-(1 + _) for _ in range(lt.parent.dimension)]
    return base_norm(lt, running_mean, running_var, weight, bias, use_input_stats, momentum, eps, dims)
