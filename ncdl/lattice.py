"""
This is largely a rework of the ideas within some of the open research I did at Huawei
in 2021. Most of the ideas in that work we're not well tested, and some of the implementation
was a little sloppy (this is all on me).

"""
from ncdl.utils import get_coset_vector_from_name
from typing import List, Tuple, Union
import torch


class LatticeTensor:
    """
    A LatticeTensor container abstracts the idea of storing information on a lattice.
    """
    def __init__(self, lt: "LatticeTensor" = None, alt_cosets: torch.Tensor = None):

        # If we have an alternative set of cosets we want to construct
        # ensure that they have the same shape
        if alt_cosets is not None and lt is not None:
            assert all([a.size() == b.size() for (a, b) in zip(lt._cosets, alt_cosets)])

        if lt is not None:
            print("copy?")

        self._cosets = []
        self._coset_offsets = []

        raise NotImplementedError("Not finished")

    def detach(self) -> "LatticeTensor":
        return LatticeTensor(self, alt_cosets=[_.detach() for _ in self._cosets])

    def clone(self) -> "LatticeTensor":
        return LatticeTensor(self, alt_cosets=[_.clone() for _ in self._cosets])

    def to(self, device) -> "LatticeTensor":
        return LatticeTensor(self, alt_cosets=[_.to(device) for _ in self._cosets])

    def is_tensor(self) -> bool:
        return len(self._cosets) == 1

    def __cmp__(self, other):
        raise NotImplementedError()

    def __getitem__(self):
        """ this is going to be a doozy --- the syntax I'm thinking right now is either
        lt[coset(int), batch slice, rectangular region on the coset] -> tensor
          or
        lt[batch slice, rectangular region in space of integers] -> LatticeTensor
        """
        raise NotImplementedError()

    def coset_vector(self, coset: int) -> torch.IntTensor:
        raise NotImplementedError()


class Lattice:
    """
    The general "LatticeTensor" factory. Basically, this holds all the important information about the
    point structure of the Lattice. LatticeTensors are instances of multi-dimensional sequences, but they
    store values only within a bounded region (and on an integer lattice).


    """
    def __init__(self,
                 input_lattice: Union[List, str],
                 scale: Union[torch.IntTensor, None] = None,
                 tensor_backend: torch.Tensor = torch.Tensor):
        """

        :param input_lattice:
        :param scale: Matrix diagonal
        :param tensor_backend:
        """
        coset = input_lattice
        if isinstance(input_lattice, str) and scale is None:
            coset, scale = get_coset_vector_from_name(input_lattice)

        if not isinstance(coset, list):
            raise ValueError(f"The input 'input_lattice' should either be a common lattice identifier "
                             f"or a list of IntTensors on the CPU that represent the coset structure "
                             f"of the lattice. If you specified the latter case, then you should also "
                             f"pass 'scale' as matrix (torch.IntTensor) that describes the scale of the "
                             f"cosets.")

        if len(coset) == 0:
            raise ValueError(f"The input 'coset' should be a non-empty list of IntTensors")

        if any([not isinstance(_, torch.IntTensor) for _ in coset]):
            raise ValueError(f"The input 'coset' should be a non-empty list of IntTensors")

        if any([_.size() != coset[0].size() for _ in coset]):
            raise ValueError(f"The input 'coset' should be a non-empty list of 1-D IntTensors with the same size")

        self.tensor = tensor_backend
        self._dimension = coset[0].shape[0]
        self._coset_scale = scale
        with torch.no_grad():
            canonical_offsets = [_ % scale for _ in coset]

        self._coset_vectors = canonical_offsets

    @property
    def dimension(self) -> int:
        return self._dimension

    @property
    def coset_scale(self) -> torch.IntTensor:
        return self._coset_scale.detach()

    @property
    def coset_count(self) -> int:
        return len(self._coset_vectors)

    def coset_offset(self, idx):
        if 0 > idx <= self.coset_count:
            raise ValueError('Invalid coset offset index!')
        return self._coset_vectors[idx].detach()

    def __cmp__(self, other):
        raise NotImplementedError()

    def __call__(self, *args):
        raise NotImplementedError()


class HalfLattice(Lattice):
    def __init__(self,
                 input_lattice: Union[List, str],
                 scale: Union[torch.FloatTensor, None] = None):
        super(HalfLattice, self).__init__(input_lattice, scale, torch.HalfTensor)


class FloatLattice(Lattice):
    def __init__(self,
                 input_lattice: Union[List, str],
                 scale: Union[torch.FloatTensor, None] = None):
        super(FloatLattice, self).__init__(input_lattice, scale, torch.FloatTensor)


class DoubleLattice(Lattice):
    def __init__(self,
                 input_lattice: Union[List, str],
                 scale: Union[torch.FloatTensor, None] = None):
        super(DoubleLattice, self).__init__(input_lattice, scale, torch.DoubleTensor)


class IntLattice(Lattice):
    def __init__(self,
                 input_lattice: Union[List, str],
                 scale: Union[torch.FloatTensor, None] = None):
        super(IntLattice, self).__init__(input_lattice, scale, torch.IntTensor)