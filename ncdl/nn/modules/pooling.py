import torch
import math
from torch import nn
from torch.nn import init
from ncdl.lattice import LatticeTensor, Lattice
from ncdl.nn.functional.pooling import lattice_maxpool


class LatticeMaxPooling(nn.Module):
    """
    LatticeMaxPooling

    """

    def __init__(self,
                 lattice: Lattice,
                 stencil: "Stencil"):
        super(LatticeMaxPooling, self).__init__()
        self.stencil = stencil
        self.lattice = lattice

        assert stencil.lattice == self.lattice

    def forward(self, lt: LatticeTensor) -> LatticeTensor:
        return lattice_maxpool(lt, self.lattice, self.stencil)