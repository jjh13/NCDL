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
    def __init__(self, lt: "LatticeTensor" = None):
        if lt is not None:
            print("copy?")
        raise NotImplementedError()

    def detach(self):
        raise NotImplementedError()

    def clone(self):
        raise NotImplementedError()

    def to(self, device):
        raise NotImplementedError()

    def is_tensor(self):
        raise NotImplementedError()

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
                 scale: Union[torch.FloatTensor, None] = None,
                 tensor_backend: torch.Tensor = torch.FloatTensor):
        coset = input_lattice
        if isinstance(input_lattice, str) and scale is None:
            coset, scale = get_coset_vector_from_name(input_lattice)

        if not isinstance(coset, list):
            raise ValueError(f"The input 'input_lattice' should either be a common lattice identifier "
                             f"or a list of IntTensors on the CPU that represent the coset structure "
                             f"of the lattice. If you specified the latter case, then you should also "
                             f"pass 'scale' as matrix (torch.IntTensor) that describes the scale of the "
                             f"cosets.")

        self.tensor = tensor_backend
        raise NotImplementedError()

    def __cmp__(self, other):
        raise NotImplementedError()

    def __call__(self, *args):
        raise NotImplementedError()
