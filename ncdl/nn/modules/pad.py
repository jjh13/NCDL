
import torch
from torch import nn
from ncdl.lattice import Lattice, LatticeTensor
from ncdl.nn.functional.downsample import downsample_lattice, downsample
from ncdl.nn.functional.upsample import upsample_lattice, upsample

class LatticePad(nn.Module):
    def __init__(self, lattice: Lattice, stencil: "Stencil"):
        super().__init__()
        self._lattice = lattice
        self._stencil = stencil

    @property
    def lattice(self):
        return self._lattice

    def forward(self, lt: LatticeTensor) -> LatticeTensor:
        if lt.parent != self._lattice:
            raise ValueError(f"Input lattice tensor does not belong to the parent lattice!")
        return self._stencil.pad_lattice_tensor(lt)
