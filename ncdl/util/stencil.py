"""
stencil.py

This file contains the astraction for "stencils" which describe the explicit geometry of filters/neighborhoods.
This is useful for any algorithm that operates over a fixed neighborhood of lattice points.
"""
import torch
import numpy as np

from math import prod
from collections.abc import Iterable
from typing import List, Tuple, Optional, Dict

from ncdl.lattice import Lattice, LatticeTensor
from ncdl.nn.functional.pad import pad, padding_for_stencil


class Stencil:
    """
    Stencil -- a container class for filters. These specify the exact geometry that convolution filters take.
    """
    def __init__(self, stencil: List[Tuple], lattice: Lattice, center: Optional[Tuple] = None):
        """
        Wraps a list of stencil points and validates that it belongs to a given lattice structure.

        :param stencil: The input list of points. Input points must all be >= 0.
        :param lattice: The input lattice factory instance. Stencil points must belong on this lattice.
        """
        bad_stencil = False
        bad_stencil |= not (isinstance(stencil, list) or isinstance(stencil, tuple))

        # Check if each entry of the stencil is iterable
        if not bad_stencil:
            bad_stencil |= any([not isinstance(_, Iterable) or len(_) != lattice.dimension for _ in stencil])

        if bad_stencil:
            raise TypeError(f"Stencil must be a list of tuples. Each tuple in the list must have "
                            f"dimension {lattice.dimension}")

        self._center = center
        self.lattice = lattice
        self.stencil = [_[:] for _ in stencil]
        self._validate_stencil()

        self._cache_cartesian_coset_stencils = None
        self._cache_coset_stencils = None
        self._cache_squareness = {}

    def _validate_stencil(self):
        """
        Validates that the stencil is correct

        TODO: Validate and/or complain

        """
        coset_mins = [[0 for _ in range(self.lattice.dimension)] for __ in range(self.lattice.coset_count)]

        for _ in self.stencil:
            coset_index, cartesian_pt = self.lattice.cartesian_index(_)

            if coset_index is None:
                raise ValueError(f"Stencil point {_} does not belong to the lattice")

            if any([(_ < 0) for _ in cartesian_pt]):
                raise ValueError(f"Stencil point {_} is negative!")

            # Take the min for the offset vectory
            for idx, value in enumerate(coset_mins[coset_index]):
                if cartesian_pt[idx] <= coset_mins[coset_index][idx]:
                    coset_mins[coset_index][idx] = value

        self.offset_vectors = [
            self.lattice.cartesian_to_lattice(pt, coset_index) for coset_index, pt in enumerate(coset_mins)
        ]

    def delta_shift(self,
                    lt: LatticeTensor,
                    coset_i: int,
                    coset_j: int) -> Tuple[int]:
        """
        Returns the appropriate \\delta shift from the paper. This value shifts the cosets of the filter so as to be
        appropriate/compatible with the coset we're convolving with.

        :param lt: An input LatticeTensor
        :param coset_i: Coset index of the output lattice
        :param coset_j: Coset index of the filter

        :returns: a tuple/list of ints rep
        """
        if lt.parent != self.lattice:
            raise ValueError()

        # This is from the paper
        kappa = self.lattice.kappa(coset_i, coset_j, additive=True)
        ds = self.offset_vectors[coset_j] + lt.coset_vectors[coset_i]
        ds = ds - lt.coset_vectors[kappa]
        scale = 1. / self.lattice.coset_scale

        mins, _ = self.stencil_boundaries(coset_j, canonical=True, cartesian=True)
        shift = tuple([-int(_.item()) for _ in ds * scale])
        shift = tuple([_ - m for _, m in zip(shift, mins)])

        # TODO: If the coset is shifted, then this should be shifted, too
        return shift

    def pad_lattice_tensor(self,
                           lt: LatticeTensor,
                           mode: str='zero',
                           value: float=0.0) -> LatticeTensor:
        """
        Utility method to appropriately pad a lattice tensor.

        :param lt: Input lattice tensor to pad
        :param mode: Padding mode, see torch.nn.functional.pad for details
        :param mode: Padding value, see torch.nn.functional.pad for details

        :returns: Returns a padded lattice tensor
        """
        padding = self.padding_for_lattice_tensor(lt)
        return pad(lt, padding, mode, value)

    def padding_for_lattice_tensor(self, lt: LatticeTensor) -> List[int]:
        """
        Utility method to calculate lattice tensor padding.

        :param lt: Input lattice tensor to calculate padding for

        :returns: Returns a list of integers that describes the padding for ncdl.nn.functional.pad.
        """
        return padding_for_stencil(lt, self.stencil, self._center)

    def weight_index(self, coset: int) -> Tuple[List[int], Dict]:
        """
        Gets an index into a weight parameter tensor.

        :param coset: The filter coset weight tensor should be for
        :returns: A tuple (filter_index, reverse_index)

        filter_index is used by unpack_weights to unpack a parameter tensor into a tensor compatible with PyTorch's
        convolution functions.

        reverse_index gives an index into a weight tensor. I.e. if I want to set stencil element (x,y) to some value, I
        would set  param_tensor[0,0, reverse_index[(x,y)]] = value.
        """
        coset_stencil = self.coset_decompose(packed_output=True)[coset]
        starts, boundaries = self.stencil_boundaries(coset, canonical=True, cartesian=True)

        # boundaries = [b-a for b,a in zip(boundaries, starts)]
        filter_index = [0] * prod(boundaries)


        reverse_index = {}
        for _, idx in enumerate(coset_stencil):
            idx = tuple([b-a for b,a in zip(idx, starts)])
            filter_index[np.ravel_multi_index(idx, boundaries)] = _ + 1
            reverse_index[idx] = _
        return filter_index, reverse_index

    def zero_weights(self,
                     coset: int,
                     channels_in: int,
                     channels_out: int,
                     device: Optional[torch.device]=None)->torch.tensor:
        """
        Creates an empty tensor of weights.

        :param coset: The filter coset weight tensor should be for
        :param channels_in: The number of channels in for the filter
        :param channels_out: The number of output channels for the filter
        :param device: The device on which the weights exist
        """
        coset_stencil = self.coset_decompose(packed_output=True)[coset]
        return torch.zeros(channels_in, channels_out, len(coset_stencil), device=device)

    def unpack_weights(self,
                       coset: int,
                       weights: torch.Tensor,
                       indices: Optional[torch.IntTensor]) -> torch.Tensor:
        """
        Upacks a weight tensor.

        :param coset: coset
        :param weights: A tensor of [channels_in, channels_out, coset_stencil_size]
        :param indices: Index `filter_index` from

        :return: a tensor of dimension [channels_in, channels_out, h, w, ...]
        """
        device = weights.device
        c_in, c_out, _ = weights.shape
        _, boundary = self.stencil_boundaries(coset, canonical=True, cartesian=True)
        if indices is not None:
            z = torch.zeros(c_in, c_out, 1, device=device)
            weights = torch.index_select(torch.cat([z, weights], dim=-1), -1, indices)
        return weights.reshape(*(c_in, c_out, *boundary))

    def coset_decompose(self, packed_output: bool = True) -> List:
        """
        This function partitions the stencil into 'n' sets, where 'n' is the number of Cartesian
        cosets of the parent lattice.

        :param packed_output: If true, each coset is scaled and shifted to the origin

        :returns: a list of lists. Each sublist contains the stencil points for the corresponding
        coset of the stencil.
        """
        if self._cache_cartesian_coset_stencils is not None and packed_output:
            return self._cache_cartesian_coset_stencils

        if self._cache_coset_stencils is not None and not packed_output:
            return self._cache_coset_stencils

        cosets = [[] for _ in range(self.lattice.coset_count)]
        for lattice_site in self.stencil:
            index = self.lattice.coset_index(lattice_site)
            if index is None:
                raise ValueError(f"Stencil contains an invalid lattice site, {lattice_site} is "
                                 f"not on the target lattice!")
            lattice_site = np.array(lattice_site, dtype='int')
            if packed_output:
                lattice_site = (lattice_site - self.lattice.coset_offset(index))//self.lattice._coset_scale

            cosets[index] += [tuple([int(_.item()) for _ in lattice_site])]

        if packed_output:
            self._cache_cartesian_coset_stencils = [_[:] for _ in cosets]
        else:
            self._cache_coset_stencils = [_[:] for _ in cosets]

        return cosets

    def is_coset_square(self, coset_index) -> bool:
        """
        Testes whether a coset of a stencil is square (i.e. packed)

        :param coset_index:

        :returns: True if coset coset_index is completely packed (has no zero entries)
        """

        if coset_index not in self._cache_squareness:
            stencil = self.coset_decompose(packed_output=True)[coset_index]
            packed = list(zip(*stencil))
            boundary_min = [min(_) for _ in packed]
            boundary_max = [max(_) for _ in packed]
            size = [b - a + 1 for b, a in zip(boundary_max, boundary_min)]

            # Cache the computed truthiness
            self._cache_squareness[coset_index] = len(stencil) == prod(size)

        return self._cache_squareness[coset_index]

    def stencil_boundaries(self,
                           coset_index: Optional[int],
                           canonical: bool = True,
                           cartesian: bool = True) -> Tuple[List[int], List[int]]:
        """
        Returns the boundaries of the stencil.

        This function can get a little complicated, so here's the breakdown. If coset_index is None, then the boundary
        computation happens all over the whole stencil. If coset_index is not None, then the boundary will be computed
        over the elements that belong to that coset.

        If canonical is


        """
        if coset_index is None:
            assert not cartesian
            coset = self.stencil[:] + [[0] * self.lattice.dimension]
        else:
            coset = self.coset_decompose(packed_output=cartesian)[coset_index]

        if canonical:
            mins = [min(_) for _ in zip(*coset)]
            coset = [[a-b for a,b in zip(_, mins)] for _ in coset]
            maxs = [max(_)+1 for _ in zip(*coset)]
            return mins, maxs

        raise NotImplementedError
