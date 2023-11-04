import torch
import torch.nn as nn
import ncdl
from ncdl.lattice import LatticeTensor, Lattice
from ncdl.util.slice import *
# from ncdl.util.stencil import Stencil
from typing import List


def lattice_conv(lt: LatticeTensor,
                 lattice: Lattice,
                 stencil: "Stencil",
                 weights,
                 bias=None,
                 groups: int = 1,
                 alt_cosets: List[torch.Tensor] = None):
    """
    This is one of the main contributions of the paper.

    """
    output_cosets = []
    batch_size = lt.coset(0).shape[0]
    channels_out = weights[0].shape[1]

    for k in range(lattice.coset_count):
        coset_output_shape = [batch_size, channels_out] + _calc_shape_out(lt, k, stencil)
        coset_result = None

        for i in range(lattice.coset_count):
            if weights[i] is  None:
                continue

            delta = stencil.delta_shift(lt, k, i)
            kappa = lt.parent.kappa(k, i)
            coset = shift_coset(lt.coset(kappa) if alt_cosets is None else alt_cosets[kappa], delta)

            partial_conv = _conv_coset(lattice, coset, weights[i], groups)

            # Enforce that the coset output is the correct size
            partial_conv = pad_or_slice_like_shape(partial_conv, coset_output_shape)

            # If we haven't started accumulating coset results
            if coset_result is None:
                coset_result = partial_conv
            else:
                summation_slice = common_tensor_slice(partial_conv, coset_result)
                coset_result[summation_slice] += partial_conv[summation_slice]

        if bias is not None:
            reshape = [1, bias.shape[0]] + [1] * lt.parent.dimension
            coset_result += bias.reshape(*reshape)

        output_cosets += [coset_result]

    return LatticeTensor(None, parent=lt.parent, alt_cosets=output_cosets, alt_offsets=lt.coset_vectors)


def _calc_shape_out(lt: LatticeTensor,
                    coset_index: int,
                    stencil: "Stencil"):
    start = lt.coset_vectors[coset_index]
    bound = lt.lattice_bounds()[2:]
    stencil_bdry = stencil.stencil_boundaries(coset_index=None, canonical=True, cartesian=False)[-1]

    shape = []
    for idx in range(lt.parent.dimension):
        l = bound[idx][1] - start[idx] + 1
        k = stencil_bdry[idx]
        s = lt.parent.coset_scale[idx].item()
        w = (l - (k - 1) - 1) // s + 1
        shape += [w]

    return [int(_.item()) for _ in shape]


def _conv_coset(_lattice, data, weights, groups):
    assert _lattice.dimension in [1, 2, 3]
    convolution = {1: nn.functional.conv1d, 2: nn.functional.conv2d, 3: nn.functional.conv3d}
    return convolution[_lattice.dimension](data, weights, padding=0, groups=groups)
