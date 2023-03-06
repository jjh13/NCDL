import torch
import ncdl

import torch.nn as nn
import ncdl.nn as ncnn

import numpy as np
from typing import Optional, Callable


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
                 mid_channels: Optional[int] = None,
                 block_type: str = 'basic',
                 normalization: Optional[str] = None,
                 skip_first: bool = False,
                 activation_function: Callable = ncnn.ReLU):
        super(ConvBlock, self).__init__()
        assert block_type in ['basic', 'unet', 'residual']
        if lattice_in == ncdl.Lattice("qc"):
            stencil = ncdl.Stencil([
                (0, 0), (2, 0), (0, 2), (2, 2), (1, 1)
            ], lattice_in, center=(1,1))
            skip_stencil = ncdl.Stencil([
                (0, 0),
            ], lattice_in, center=(0,0))
        else:
            stencil = ncdl.Stencil([
                (0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2), (2, 0), (2, 1), (2, 2)
            ], lattice_in, center=(1,1))
            skip_stencil = ncdl.Stencil([
                (0, 0),
            ], lattice_in, center=(0,0))

        self.skip_path = None
        mid_channels = mid_channels if mid_channels else channels_out
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
                ncnn.LatticeConvolution(lattice_in, channels_in=channels_in, channels_out=mid_channels, stencil=stencil),
                LatticeNormalizationWrapper(lattice_in, mid_channels, normalization),
                activation_function(),

                ncnn.LatticePad(lattice_in, stencil),
                ncnn.LatticeConvolution(lattice_in, channels_in=mid_channels, channels_out=channels_out, stencil=stencil),
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
                 activation_function: Callable = ncnn.ReLU,
                 resample_rate: int = 2):
        """

        :param lattice_in: Input tensors for the forward function expect data on this lattice
        """
        super(DownBlock, self).__init__()
        assert resample_rate in [1, 2, 4]

        resample_matrix = {
            1: [[1, 0], [0, 1]],
            2: [[1, 1], [1, -1]],
            4: [[2, 0], [0, 2]]
        }
        self.lattice_in = lattice_in
        self.block = ConvBlock(lattice_in,
                               channels_in=channels_in,
                               channels_out=channels_out,
                               block_type=block_type,
                               normalization=normalization,
                               skip_first=skip_first,
                               activation_function=activation_function)

        self.downsample = ncnn.LatticeDownsample(lattice_in, np.array(
            resample_matrix[resample_rate],
        dtype='int'))

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
                 activation_function: Callable = ncnn.ReLU,
                 resample_rate: int = 2):
        """

        :param lattice_in: Input tensors for the forward function expect data on this lattice
        """
        super(UpBlock, self).__init__()
        assert resample_rate in [1, 2, 4]

        resample_matrix = {
            1: [[1, 0], [0, 1]],
            2: [[1, 1], [1, -1]],
            4: [[2, 0], [0, 2]]
        }
        self.lattice_in = lattice_in
        self.upsample = ncnn.LatticeUpsample(lattice_in, np.array(
            resample_matrix[resample_rate],
        dtype='int'))
        self.skip_input = skip_input
        self.low_pass = None
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


class InnerRecursiveNetwork(nn.Module):
    def __init__(self,
                 lattice_in,
                 structure,
                 inner_channels: Optional[int] = None,
                 middle_channels: Optional[int] = None,
                 block_type: str = 'basic',
                 normalization: str = 'bn',
                 activation_function: Callable = ncnn.ReLU):
        super(InnerRecursiveNetwork, self).__init__()

        if len(structure) == 0:
            print('!!!')
            self.down_block = None
            self.up_block = None
            self.inner_block = ConvBlock(
                lattice_in,
                channels_in=inner_channels,
                mid_channels=middle_channels,
                channels_out=inner_channels,
                block_type=block_type,
                normalization=normalization,
                activation_function=activation_function
            )
        else:
            channels_in, channels_out, down_sample, output_override, skip, \
            norm, activation, nohead, block_type = structure[0]
            structure = structure[1:]

            self.down_block = DownBlock(
                lattice_in,
                channels_in,
                channels_out,
                block_type,
                norm,
                nohead,
                activation,
                down_sample
            )

            self.inner_block = InnerRecursiveNetwork(
                self.down_block.lattice_out,
                structure,
                inner_channels = inner_channels,
                middle_channels = middle_channels,
                block_type = block_type,
                normalization = normalization,
                activation_function = activation_function)

            true_output = output_override if output_override else channels_in
            self.up_block = UpBlock(
                self.down_block.lattice_out,
                channels_out,
                true_output,
                channels_out,
                block_type,
                norm,
                False,
                activation,
                down_sample,

            )

    def forward(self, lt: ncdl.LatticeTensor):
        if self.down_block:
            lt, lt_down = self.down_block(lt)
        else:
            lt, lt_down = None, lt

        lt_down = self.inner_block(lt_down)

        if self.up_block:
            return self.up_block(lt_down, lt)
        return lt_down


class Unet(nn.Module):
    def __init__(self, channels_in, channels_out, config, residual: bool = False):
        super().__init__()
        lattice_cp = ncdl.Lattice('cp')
        self.lattice = lattice_cp

        normalization = 'bn' if residual else None
        block_type = 'unet' if not residual else 'residual'

        if config == 'ae_cp_std':
            self.inner_network = InnerRecursiveNetwork(
                lattice_cp,
                [(channels_in, 64, 4,   64, False, None, ncnn.ReLU, True, 'basic'),
                 (64,         128, 4, None, False, None, ncnn.ReLU, False, 'basic'),
                 (128,        256, 4, None, False, None, ncnn.ReLU, False, 'basic'),
                 (256,        512, 4, None, False, None, ncnn.ReLU, False, 'basic')],
                inner_channels=512,
                middle_channels=1024
            )
            self.epilogue = nn.Conv2d(64, channels_out, kernel_size=1, padding=0)
        elif config == 'ae_cp_dbl':
            self.inner_network = InnerRecursiveNetwork(
                lattice_cp,
                [(channels_in, 64, 4,   64, False, None, ncnn.ReLU, True, 'basic'),
                 (64,          64, 1, None, False, None, ncnn.ReLU, True, 'basic'),

                 (64,         128, 4, None, False, None, ncnn.ReLU, False, 'basic'),
                 (128,        128, 1, None, False, None, ncnn.ReLU, True, 'basic'),

                 (128,        256, 4, None, False, None, ncnn.ReLU, False, 'basic'),
                 (256,        256, 1, None, False, None, ncnn.ReLU, False, 'basic'),

                 (256,        512, 4, None, False, None, ncnn.ReLU, False, 'basic'),
                 (512,        512, 1, None, False, None, ncnn.ReLU, False, 'basic')],
                inner_channels=512,
                middle_channels=1024
            )
            self.epilogue = nn.Conv2d(64, channels_out, kernel_size=1, padding=0)
        elif config == 'ae_qc_dbl':
            self.inner_network = InnerRecursiveNetwork(
                lattice_cp,
                [(channels_in, 64, 2, 64, False, None, ncnn.ReLU, True, 'basic'),
                 (64,   64, 2, None, False, None, ncnn.ReLU, True, 'basic'),

                 (64,  128, 2, None, False, None, ncnn.ReLU, False, 'basic'),
                 (128, 128, 2, None, False, None, ncnn.ReLU, True, 'basic'),

                 (128, 256, 2, None, False, None, ncnn.ReLU, False, 'basic'),
                 (256, 256, 2, None, False, None, ncnn.ReLU, False, 'basic'),

                 (256, 512, 2, None, False, None, ncnn.ReLU, False, 'basic'),
                 (512, 512, 2, None, False, None, ncnn.ReLU, False, 'basic')],
                inner_channels=512,
                middle_channels=1024
            )
            self.epilogue = nn.Conv2d(64, channels_out, kernel_size=1, padding=0)
        elif config == 'ae_qc_grdl':
            self.inner_network = InnerRecursiveNetwork(
                lattice_cp,
                [(channels_in, 64, 2, 64, False, None, ncnn.ReLU, True, 'basic'),
                 (64, 96, 2, None, False, None, ncnn.ReLU, False, 'basic'),

                 (96, 128, 2, None, False, None, ncnn.ReLU, False, 'basic'),
                 (128, 192, 2, None, False, None, ncnn.ReLU, False, 'basic'),

                 (192, 256, 2, None, False, None, ncnn.ReLU, False, 'basic'),
                 (256, 384, 2, None, False, None, ncnn.ReLU, False, 'basic'),

                 (384, 512, 2, None, False, None, ncnn.ReLU, False, 'basic'),
                 (512, 512, 2, None, False, None, ncnn.ReLU, False, 'basic')],
                inner_channels=512,
                middle_channels=1024
            )
            self.epilogue = nn.Conv2d(64, channels_out, kernel_size=1, padding=0)
        elif config == 'ae_cp_grdl':
            self.inner_network = InnerRecursiveNetwork(
                lattice_cp,
                [(channels_in, 64, 4, 64, False, None, ncnn.ReLU, True, 'basic'),
                 (64, 96, 1, None, False, None, ncnn.ReLU, False, 'basic'),

                 (96, 128, 4, None, False, None, ncnn.ReLU, False, 'basic'),
                 (128, 192, 1, None, False, None, ncnn.ReLU, False, 'basic'),

                 (192, 256, 4, None, False, None, ncnn.ReLU, False, 'basic'),
                 (256, 384, 1, None, False, None, ncnn.ReLU, False, 'basic'),

                 (384, 512, 4, None, False, None, ncnn.ReLU, False, 'basic'),
                 (512, 512, 1, None, False, None, ncnn.ReLU, False, 'basic')],
                inner_channels=512,
                middle_channels=1024
            )
            self.epilogue = nn.Conv2d(64, channels_out, kernel_size=1, padding=0)

        elif config == 'unet_cp_std':
            self.inner_network = InnerRecursiveNetwork(
                lattice_cp,
                [(channels_in, 64, 4,   64, True, normalization, ncnn.ReLU, True, block_type),
                 (64,         128, 4, None, True, normalization, ncnn.ReLU, False, block_type),
                 (128,        256, 4, None, True, normalization, ncnn.ReLU, False, block_type),
                 (256,        512, 4, None, True, normalization, ncnn.ReLU, False, block_type)],
                inner_channels=512,
                middle_channels=1024
            )
            self.epilogue = nn.Conv2d(64, channels_out, kernel_size=1, padding=0)
        elif config == 'unet_cp_dbl':
            self.inner_network = InnerRecursiveNetwork(
                lattice_cp,
                [(channels_in, 64, 4,   64, False, normalization, ncnn.ReLU, True, block_type),
                 (64,          64, 1, None, True, normalization, ncnn.ReLU, False, block_type),

                 (64,         128, 4, None, False, normalization, ncnn.ReLU, False, block_type),
                 (128,        128, 1, None, True, normalization, ncnn.ReLU, False, block_type),

                 (128,        256, 4, None, False, normalization, ncnn.ReLU, False, block_type),
                 (256,        256, 1, None, True, normalization, ncnn.ReLU, False, block_type),

                 (256,        512, 4, None, False, normalization, ncnn.ReLU, False, block_type),
                 (512,        512, 1, None, True, normalization, ncnn.ReLU, False, block_type)
                 ],
                inner_channels=512,
                middle_channels=1024
            )
            self.epilogue = nn.Conv2d(64, channels_out, kernel_size=1, padding=0)

        elif config == 'unet_qc_dbl':
            self.inner_network = InnerRecursiveNetwork(
                lattice_cp,
                [(channels_in, 64, 2, 64, False, normalization, ncnn.ReLU, True, block_type),
                 (64, 64, 2, None, True, normalization, ncnn.ReLU, False, block_type),

                 (64, 128, 2, None, False, normalization, ncnn.ReLU, False, block_type),
                 (128, 128, 2, None, True, normalization, ncnn.ReLU, False, block_type),

                 (128, 256, 2, None, False, normalization, ncnn.ReLU, False, block_type),
                 (256, 256, 2, None, True, normalization, ncnn.ReLU, False, block_type),

                 (256, 512, 2, None, False, normalization, ncnn.ReLU, False, block_type),
                 (512, 512, 2, None, False, normalization, ncnn.ReLU, False, block_type)],
                inner_channels=512,
                middle_channels=1024
            )
            self.epilogue = nn.Conv2d(64, channels_out, kernel_size=1, padding=0)
        elif config == 'unet_cp_grdl':
            self.inner_network = InnerRecursiveNetwork(
                lattice_cp,
                [(channels_in, 64, 4,   64, False, normalization, ncnn.ReLU, True, block_type),
                 (64,          96, 1, None, True, normalization, ncnn.ReLU, False, block_type),

                 (96,         128, 4, None, False, normalization, ncnn.ReLU, False, block_type),
                 (128,        192, 1, None, True, normalization, ncnn.ReLU, False, block_type),

                 (192,        256, 4, None, False, normalization, ncnn.ReLU, False, block_type),
                 (256,        384, 1, None, True, normalization, ncnn.ReLU, False, block_type),

                 (384,        512, 4, None, False, normalization, ncnn.ReLU, False, block_type),
                 (512,        512, 1, None, True, normalization, ncnn.ReLU, False, block_type)
                 ],
                inner_channels=512,
                middle_channels=1024
            )
            self.epilogue = nn.Conv2d(64, channels_out, kernel_size=1, padding=0)

        elif config == 'unet_qc_grdl':
            self.inner_network = InnerRecursiveNetwork(
                lattice_cp,
                [(channels_in, 64, 2, 64, False, normalization, ncnn.ReLU, True, block_type),
                 (64, 96, 2, None, True, normalization, ncnn.ReLU, False, block_type),

                 (96, 128, 2, None, False, normalization, ncnn.ReLU, False, block_type),
                 (128, 192, 2, None, True, normalization, ncnn.ReLU, False, block_type),

                 (192, 256, 2, None, False, normalization, ncnn.ReLU, False, block_type),
                 (256, 384, 2, None, True, normalization, ncnn.ReLU, False, block_type),

                 (384, 512, 2, None, False, normalization, ncnn.ReLU, False, block_type),
                 (512, 512, 2, None, False, normalization, ncnn.ReLU, False, block_type)],
                inner_channels=512,
                middle_channels=1024
            )
            self.epilogue = nn.Conv2d(64, channels_out, kernel_size=1, padding=0)

    def forward(self, x):
        lt = self.lattice(x)
        lt = self.inner_network(lt)
        return self.epilogue(lt.coset(0))
