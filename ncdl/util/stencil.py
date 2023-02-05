from collections.abc import Iterable
from typing import List, Tuple, Optional
from math import prod
from ncdl.lattice import Lattice, LatticeTensor
import torch


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


        self.lattice = lattice
        self.stencil = [_[:] for _ in stencil]
        self._validate_stencil()

        self._cache_cartesian_coset_stencils = None
        self._cache_coset_stencils = None
        self._cache_squareness = {}

    def _validate_stencil(self):
        coset_mins = [[0 for _ in range(self.lattice.dimension)] for __ in range(self.lattice.coset_count)]

        for _ in self.stencil:
            coset_index, cartesian_pt = self.lattice.cartesian_index(_)

            if coset_index is None:
                raise ValueError(f"Stencil point {_} does not belong to the lattice")

            if any([(_ < 0) if coset_index == 0 else (_ < -1) for _ in cartesian_pt]):
                raise ValueError(f"Stencil point {_} is outside the canonical region defined by the lattice")

            # Take the min for the offset vectory
            for idx, value in enumerate(coset_mins[coset_index]):
                if cartesian_pt[idx] <= coset_mins[coset_index][idx]:
                    coset_mins[coset_index][idx] = value

        self.offset_vectors = [
            self.lattice.cartesian_to_lattice(pt, coset_index) for coset_index, pt in enumerate(coset_mins)
        ]

    def delta_shift(self, lt: LatticeTensor, coset_i, coset_j):
        # This is from the paper
        kappa = self.lattice.kappa(coset_i, coset_j, additive=True)
        ds = self.offset_vectors[coset_j] + lt.coset_vectors[coset_i]
        ds = ds - lt.coset_vectors[kappa]
        scale = 1. / self.lattice.coset_scale
        return [-int(_.item()) for _ in ds * scale]

    def pad_lattice_tensor(self, lt, mode='zero', value=0.0):
        pass

    def padding_for_lattice_tensor(self, lt):
        pass

    def weight_index(self, coset):
        import numpy as np

        coset_stencil = self.coset_decompose(output_cartesian=True)[coset]
        _, boundaries = self.stencil_boundaries(coset, canonical=True, cartesian=True)
        filter_index = [0] * prod(boundaries)
        reverse_index = {}
        for _, idx in enumerate(coset_stencil):
            filter_index[np.ravel_multi_index(idx, boundaries)] = _ + 1
            reverse_index[idx] = _
        return filter_index, reverse_index

    def zero_weights(self, coset, channels_in, channels_out, device=None):
        coset_stencil = self.coset_decompose(output_cartesian=True)[coset]
        return torch.zeros(channels_in, channels_out, len(coset_stencil), device=device)

    def unpack_weights(self, coset, weights, indices):
        device = weights.device
        c_in, c_out, _ = weights.shape
        _, boundary = self.stencil_boundaries(coset, canonical=True, cartesian=True)
        if indices is not None:
            z = torch.zeros(c_in, c_out, 1, device=device)
            weights = torch.index_select(torch.cat([z, weights], dim=-1), -1, indices)
        return weights.reshape(*(c_in, c_out, *boundary))

    def weights_to_filter(self, coset, weights):
        """ Returns a """
        pass

    def coset_decompose(self, output_cartesian=True):
        """
        This function partitions the stencil into 'n' sets, where 'n' is the number of Cartesian
        cosets of the parent lattice.

        Returns a list of lists. Each sublist contains the stencil points for the corresponding
        coset of the stencil.
        """
        if self._cache_cartesian_coset_stencils is not None and output_cartesian:
            return self._cache_cartesian_coset_stencils

        if self._cache_coset_stencils is not None and not output_cartesian:
            return self._cache_coset_stencils

        cosets = [[] for _ in range(self.lattice.coset_count)]
        for lattice_site in self.stencil:
            index = self.lattice.coset_index(lattice_site)
            if index is None:
                raise ValueError(f"Stencil contains an invalid lattice site, {lattice_site} is "
                                 f"not on the target lattice!")
            lattice_site = torch.IntTensor(lattice_site)
            if output_cartesian:
                lattice_site = (lattice_site - self.lattice.coset_offset(index))//self.lattice._coset_scale

            cosets[index] += [tuple([int(_.item()) for _ in lattice_site])]

        if output_cartesian:
            self._cache_cartesian_coset_stencils = [_[:] for _ in cosets]
        else:
            self._cache_coset_stencils = [_[:] for _ in cosets]

        return cosets

    def stencil_boundaries(self, coset_index, canonical=True, cartesian=True):
        if coset_index is None:
            coset = self.stencil + [[0] * self.lattice.dimension]
        else:
            coset = self.coset_decompose(output_cartesian=cartesian)[coset_index]

        if canonical:
            mins = [min(_) for _ in zip(*coset)]
            coset = [[a-b for a,b in zip(_, mins)] for _ in coset]
            maxs = [max(_)+1 for _ in zip(*coset)]
            return mins, maxs

        raise NotImplementedError

    def is_coset_square(self, coset_index):
        """
        Testes whether a coset of a stencil is square.
        """

        if coset_index not in self._cache_squareness:
            stencil = self.coset_decompose(output_cartesian=True)[coset_index]
            packed = list(zip(*stencil))
            boundary_min = [min(_) for _ in packed]
            boundary_max = [max(_) for _ in packed]
            size = [b - a + 1 for b, a in zip(boundary_max, boundary_min)]

            # Cache the computed truthiness
            self._cache_squareness[coset_index] = len(stencil) == prod(size)

        return self._cache_squareness[coset_index]

    def is_canonical(self):
        return True
        # """
        # REturns true
        # """
        # if self._cache_canonical is not None:
        #     return self._cache_canonical
        #
        # self._cache_canonical = True
        # for stencil in self.coset_decompose(output_cartesian=False):
        #     for site in stencil:
        #         if any([_ for _ in site < 0]):
        #             self._cache_canonical = False
        #             return self._cache_canonical
        # return self._cache_canonical

    def canonicalize(self):
        """
        Moves the stencil so that all of its locations are positive.
        """
        changed = False

        coset_stencils = self.coset_decompose(output_cartesian=False)

        if not self.is_canonical():
            pass
        # x|x x
        # go-o-o
        # We're looking for a shift f \in L that



        # Clear out stencil cache if we changed the internal representation of the stencil
        if changed:
            self._cache_coset_stencils = None
            self._cache_cartesian_coset_stencils = None
            self._cache_squareness = {}
            self._cache_canonical = True

