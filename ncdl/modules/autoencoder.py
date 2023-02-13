import torch
import torch.nn as nn

import ncdl
import ncdl.nn as ncnn


class DownBlock(nn.Module):
    def __init__(self, lattice_in: ncdl.Lattice):
        """

        :param lattice_in: Input tensors for the forward function expect data on this lattice
        """
        super(DownBlock, self).__init__()

        if lattice_in == ncdl.Lattice("qc"):
            stencil = ncdl.Stencil([
                (0, 0), (2, 0), (0, 2), (2, 2), (1, 1)
            ], lattice_in, center=(1,1))
        else:
            stencil = ncdl.Stencil([
                (0, 0), (2, 0), (0, 2), (2, 2), (1, 1)
            ], lattice_in, center=(1,1))


        self.block = nn.Sequential(
            ncnn.LatticePad(lattice_in, stencil),
            ncnn.LatticeConvolution(lattice_in, 1, 1, stencil),
            ncnn.LatticeDownsample(lattice_in)
        )


class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()



