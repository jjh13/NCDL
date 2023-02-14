
#TODO: There's a bit of redundant work going on every iteration.

import torch
from torch import nn
from ncdl.lattice import Lattice, LatticeTensor
from ncdl.nn.functional.downsample import downsample_lattice, downsample
from ncdl.nn.functional.upsample import upsample_lattice, upsample
import numpy as np


class LatticeUpsample(nn.Module):
    def __init__(self, lattice: Lattice, upsampling_matrix):
        super().__init__()

        self._lattice = lattice
        self._up_lattice = upsample_lattice(lattice, upsampling_matrix)
        self._smatrix = upsampling_matrix

    @property
    def up_lattice(self):
        return self._up_lattice

    def forward(self, lt: LatticeTensor) -> LatticeTensor:
        if lt.parent != self._lattice:
            raise ValueError(f"Input lattice tensor does not belong to the parent lattice!")
        return upsample(lt, self._smatrix)


class LatticeDownsample(nn.Module):
    def __init__(self, lattice: Lattice, downsampling_matrix):
        super().__init__()
        self._lattice = lattice
        self._down_lattice = downsample_lattice(lattice, downsampling_matrix)
        self._smatrix = downsampling_matrix

    @property
    def down_lattice(self):
        return self._down_lattice

    def forward(self, lt: LatticeTensor) -> LatticeTensor:
        if lt.parent != self._lattice:
            raise ValueError(f"Input lattice tensor does not belong to the parent lattice!")
        return downsample(lt, np.array(self._smatrix, dtype='int'))