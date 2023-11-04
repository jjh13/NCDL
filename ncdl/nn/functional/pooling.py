import torch
import torch.nn as nn
import ncdl
from ncdl.lattice import LatticeTensor, Lattice
from ncdl.util.slice import *
# from ncdl.util.stencil import Stencil
from typing import List
from ncdl.nn.functional.convolution import _calc_shape_out
from ncdl.extensions.atrous_pooling import MaxPoolAtrousFunction


def lattice_maxpool(lt: LatticeTensor, lattice: Lattice, stencil: "Stencil"):
    """
    This is one of the main contributions of the paper.
    """
    output_cosets = []
    batch_size = lt.coset(0).shape[0]
    channels_out = lt.coset(0).shape[1]

    for k in range(lattice.coset_count):
        coset_output_shape = [batch_size, channels_out] + _calc_shape_out(lt, k, stencil)
        coset_result = None

        for i in range(lattice.coset_count):
            delta = stencil.delta_shift(lt, k, i)
            kappa = lt.parent.kappa(k, i)
            coset = shift_coset(lt.coset(kappa), delta)

            if stencil.is_coset_square(i):
                _, kernel_size = stencil.stencil_boundaries(i)
                partial_pool = _max_pool_coset_builtin(lattice, coset, kernel_size)
            else:
                coset_stencil = stencil.coset_decompose(packed_output=True)[i]
                partial_pool = _max_pool_coset(lattice, coset, coset_stencil)

            # Enforce that the coset output is the correct size
            partial_pool = pad_or_slice_like_shape(partial_pool, coset_output_shape)

            # If we haven't started accumulating coset results
            if coset_result is None:
                coset_result = partial_pool
            else:
                maximation_slice = common_tensor_slice(partial_pool, coset_result)
                coset_result[maximation_slice] = torch.maximum(
                    coset_result[maximation_slice],
                    partial_pool[maximation_slice]
                )

        output_cosets += [coset_result]

    return LatticeTensor(None, parent=lt.parent, alt_cosets=output_cosets, alt_offsets=lt.coset_vectors)


def _max_pool_coset_builtin(_lattice, data, kernel_size):
    assert _lattice.dimension in [1, 2, 3]
    maxpool = {1: nn.functional.max_pool1d, 2: nn.functional.max_pool2d, 3: nn.functional.max_pool3d}
    return maxpool[_lattice.dimension](data, kernel_size=kernel_size, stride=1)


def _max_pool_coset(_lattice, data, stencil):
    assert _lattice.dimension in [1, 2, 3]
    return MaxPoolAtrousFunction.apply(data, stencil)
