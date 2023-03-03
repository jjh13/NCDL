import torch
import ncdl

import torch.nn as nn
import ncdl.nn as ncnn

import numpy as np
from typing import Optional


class LatticeNormalizationWrapper(nn.Module):
    def __init__(self, lattice, channels, normalization):
        super(LatticeNormalizationWrapper, self).__init__()
        assert normalization in [None, 'bn', 'gn', 'in']
        if normalization is None:
            self.module = nn.Identity()
        elif normalization == 'bn':
            self.module = ncnn.LatticeBatchNorm(lattice, channels)
        elif normalization == 'in':
            self.module = ncnn.LatticeInstanceNorm(lattice, channels)
        elif normalization == 'gn':
            group_size = [group_size for group_size in [8,4,2,1] if channels % group_size == 0][0]
            self.module = ncnn.LatticeGroupNorm(lattice, channels//group_size, channels)

    def forward(self, x):
        return self.module(x)


class ConvBlock(nn.Module):
    def __init__(self,
                 lattice_in: ncdl.Lattice,
                 channels_in: int,
                 channels_out: int,
                 block_type: str = 'basic',
                 normalization: Optional[str] = None,
                 skip_first: bool = False,
                 activation_function: type[nn.Module] = ncnn.ReLU):
        super(ConvBlock, self).__init__()
        assert block_type == ['basic', 'unet', 'residual']
        if lattice_in == ncdl.Lattice("qc"):
            stencil = ncdl.Stencil([
                (0, 0), (2, 0), (0, 2), (2, 2), (1, 1)
            ], lattice_in, center=(1,1))
            skip_stencil = ncdl.Stencil([
                (0, 0), (2, 0), (0, 2), (2, 2), (1, 1)
            ], lattice_in, center=(1,1))
        else:
            stencil = ncdl.Stencil([
                (0, 0), (2, 0), (0, 2), (2, 2), (1, 1)
            ], lattice_in, center=(1,1))
            skip_stencil = ncdl.Stencil([
                (0, 0), (2, 0), (0, 2), (2, 2), (1, 1)
            ], lattice_in, center=(1,1))

        self.skip_path = None

        if block_type == 'basic':
            self.conv_path = nn.Sequential(
                ncnn.LatticePad(lattice_in, stencil),
                ncnn.LatticeConvolution(lattice_in, channels_in=channels_in, channels_out=channels_out, stencil=stencil),
                LatticeNormalizationWrapper(lattice_in, channels_out, normalization),
                activation_function()
            )
        else:
            if block_type == 'residual':
                self.skip_path = nn.Sequential(
                    ncnn.LatticePad(lattice_in, skip_stencil),
                    ncnn.LatticeConvolution(lattice_in, channels_in, channels_out, skip_stencil),
                    LatticeNormalizationWrapper(lattice_in, channels_out, normalization)
                )

            layers = []
            if not skip_first:
                layers = [
                    LatticeNormalizationWrapper(lattice_in, channels_in, normalization),
                    activation_function()
                ]
            layers += [
                ncnn.LatticePad(lattice_in, stencil),
                ncnn.LatticeConvolution(lattice_in, channels_in=channels_in, channels_out=channels_out, stencil=stencil),
                LatticeNormalizationWrapper(lattice_in, channels_out, normalization),
                activation_function(),

                ncnn.LatticePad(lattice_in, stencil),
                ncnn.LatticeConvolution(lattice_in, channels_in=channels_in, channels_out=channels_out, stencil=stencil),
            ]
            self.conv_path = nn.Sequential(*layers)

    def forward(self, x):
        output = self.conv_path(x)
        if self.skip_path:
            output = output + self.skip_path(x)
        return output


class DownBlock(nn.Module):
    def __init__(self,
                 lattice_in: ncdl.Lattice,
                 channels_in: int,
                 channels_out: int,
                 block_type: str = 'basic',
                 normalization: Optional[str] = None,
                 skip_first: bool = False,
                 activation_function: type[nn.Module] = ncnn.ReLU):
        """

        :param lattice_in: Input tensors for the forward function expect data on this lattice
        """
        super(DownBlock, self).__init__()
        self.lattice_in = lattice_in
        self.block = ConvBlock(lattice_in,
                               channels_in=channels_in,
                               channels_out=channels_out,
                               block_type=block_type,
                               normalization=normalization,
                               skip_first=skip_first,
                               activation_function=activation_function)
        self.downsample = ncnn.LatticeDownsample(lattice_in, np.array([[1, 1], [1, -1]], dtype='int'))

    @property
    def lattice_out(self):
        return self.downsample.down_lattice

    def forward(self, lt: ncdl.LatticeTensor):
        lt = self.block(lt)
        return lt, self.downsample(lt)


class UpBlock(nn.Module):
    def __init__(self,
                 lattice_in: ncdl.Lattice,
                 channels_in: int,
                 channels_out: int,
                 skip_input: Optional[int] = 0,
                 block_type: str = 'basic',
                 normalization: Optional[str] = None,
                 skip_first: bool = False,
                 activation_function: type[nn.Module] = ncnn.ReLU):
        """

        :param lattice_in: Input tensors for the forward function expect data on this lattice
        """
        super(UpBlock, self).__init__()
        self.lattice_in = lattice_in
        self.upsample = ncnn.LatticeUpsample(lattice_in, np.array([[1, 1], [1, -1]], dtype='int'))
        self.skip_input = skip_input
        self.block = ConvBlock(self.upsample.up_lattice,
                               channels_in=channels_in + skip_input,
                               channels_out=channels_out,
                               block_type=block_type,
                               normalization=normalization,
                               skip_first=skip_first,
                               activation_function=activation_function)

    @property
    def lattice_out(self):
        return self.upsample.up_lattice

    def forward(self,
                lt: ncdl.LatticeTensor,
                example: Optional[ncdl.LatticeTensor] = None):
        # Input Check
        assert lt.parent == self.lattice_in
        if self.skip_input != 0:
            assert example is not None

        # Do any upsampling/Padding
        lt = self.upsample(lt)
        if example:
            assert example.parent == self.lattice_out
            lt = ncdl.pad_like(lt, example)

        # Low-passing reduces the checkerboard artifacts
        if self.low_pass:
            lt = self.low_pass(lt)

        # If we have a unet arch, then we add in the skip
        if self.skip_input != 0:
            lt = ncdl.cat([lt, example], dim=1)

        return self.block(lt)






class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()



