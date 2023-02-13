import torch
import torch.nn as nn
from ncdl.lattice import LatticeTensor, Lattice


class LatticeWrap(nn.Module):
    def __init__(self):
        super(LatticeWrap, self).__init__()

    def forward(self, tensor: torch.Tensor):

        if len(tensor.shape) == 4:
            lattice = Lattice("cp")
        elif len(tensor.shape) == 4:
            lattice = Lattice("cc")
        else:
            raise ValueError
        return lattice(tensor)


class LatticeUnwrap(nn.Module):
    def __init__(self, coset_index=0):
        super(LatticeUnwrap, self).__init__()
        self.coset_index = coset_index

    def forward(self, lt: LatticeTensor):
        return lt.coset(self.coset_index)