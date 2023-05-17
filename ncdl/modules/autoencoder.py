import torch
import ncdl
import os

import torch.nn as nn
import ncdl.nn as ncnn

import numpy as np
from typing import Optional, Callable
import torchvision
import pathlib

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


class DoubleConv(nn.Module):

    def __init__(self,
                 lattice: ncdl.Lattice,
                 in_channels: int,
                 out_channels: int,
                 mid_channels: Optional[int] =None):
        super().__init__()
        self.lattice = lattice

        if lattice == ncdl.Lattice("qc"):
            stencil = ncdl.Stencil([
                (1, 1), (2, 2), (3, 1), (1, 3), (3, 3), (0, 2), (2, 0), (2, 4), (4, 2)

            ], lattice, center=(2, 2))
        else:
            stencil = ncdl.Stencil([
                (0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2), (2, 0), (2, 1), (2, 2)
            ], lattice, center=(1, 1))

        if not mid_channels:
            mid_channels = out_channels

        self.double_conv = nn.Sequential(
            ncnn.LatticePad(lattice, stencil),
            ncnn.LatticeConvolution(lattice, channels_in=in_channels, channels_out=mid_channels, stencil=stencil, bias=False),

            # ncnn.LatticeUnwrap(),
            # nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=0, bias=False),
            # ncnn.LatticeWrap(),
            LatticeNormalizationWrapper(lattice, mid_channels, 'bn'),
            ncnn.LeakyReLU(),

            ncnn.LatticePad(lattice, stencil),
            ncnn.LatticeConvolution(lattice, channels_in=mid_channels, channels_out=out_channels, stencil=stencil, bias=False),
            # ncnn.LatticeUnwrap(),
            # nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=0, bias=False),
            # ncnn.LatticeWrap(),
            LatticeNormalizationWrapper(lattice, out_channels, 'bn'),
            ncnn.LeakyReLU(),

            # ncnn.LatticeWrap()
            # ncnn.LatticePad(lattice, stencil),
            # ncnn.LatticeConvolution(lattice, channels_in=in_channels, channels_out=mid_channels, stencil=stencil, bias=False),
            # LatticeNormalizationWrapper(lattice, mid_channels, 'bn'),
            # ncnn.LeakyReLU(),
            # ncnn.LatticePad(lattice, stencil),
            # ncnn.LatticeConvolution(lattice, channels_in=mid_channels, channels_out=out_channels, stencil=stencil, bias=False),
            # LatticeNormalizationWrapper(lattice, out_channels, 'bn'),
            # ncnn.LeakyReLU()
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self,
                 lattice: ncdl.Lattice,
                 in_channels: int,
                 out_channels: int,
                 resample_rate: int = 4):
        super().__init__()
        self.lattice = lattice

        assert resample_rate in [1, 2, 4]

        resample_matrix = {
            1: [[1, 0], [0, 1]],
            2: [[1, 1], [1, -1]],
            4: [[2, 0], [0, 2]]
        }

        self.downsample = ncnn.LatticeDownsample(lattice,
            np.array(resample_matrix[resample_rate], dtype='int')
        )

        self.conv = DoubleConv(
            self.downsample.down_lattice,
            in_channels=in_channels,
            out_channels=out_channels
        )

    @property
    def lattice_out(self):
        return self.downsample.down_lattice

    def forward(self, x):
        x = self.downsample(x)
        return self.conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, lattice, in_channels, out_channels, resample_rate=4):
        super().__init__()

        resample_matrix = {
            1: [[1, 0], [0, 1]],
            2: [[1, 1], [1, -1]],
            4: [[2, 0], [0, 2]]
        }
        self.lattice_in = lattice
        self.upsample = ncnn.LatticeUpsample(lattice, np.array(
            resample_matrix[resample_rate],
        dtype='int'))

        self.up = DoubleConv(self.upsample.up_lattice, in_channels, in_channels)
        self.conv = DoubleConv(self.upsample.up_lattice, in_channels + in_channels//2, out_channels, mid_channels=in_channels)

        # # if bilinear, use the normal convolutions to reduce the number of channels
        # if bilinear:
        #     self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        #     self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        # else:
        #     self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
        #     self.conv = DoubleConv(in_channels, out_channels)


    @property
    def lattice_out(self):
        return self.upsample.up_lattice

    def forward(self, x1: ncdl.LatticeTensor, x2: ncdl.LatticeTensor):
        x1 = self.upsample(x1)
        x1 = ncdl.pad_like(x1, x2)
        x1 = self.up(x1)
        x = ncdl.cat([x2, x1], dim=1)
        return self.conv(x)

class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Sequential(
            ncnn.LatticeUnwrap(),
            nn.Conv2d(in_channels, out_channels, kernel_size=1)
        )

    def forward(self, x):
        return self.conv(x)



class ConvBlock(nn.Module):
    def __init__(self,
                 lattice_in: ncdl.Lattice,
                 channels_in: int,
                 channels_out: int,
                 mid_channels: Optional[int] = None,
                 block_type: str = 'basic',
                 normalization: Optional[str] = None,
                 skip_first: bool = False,
                 activation_function: Callable = ncnn.LeakyReLU):
        super(ConvBlock, self).__init__()
        assert block_type in ['basic', 'unet', 'residual']
        if lattice_in == ncdl.Lattice("qc"):
            stencil = ncdl.Stencil([
                (1, 1), (2, 2), (3, 1), (1, 3), (3, 3), (0, 2), (2, 0), (2, 4), (4, 2)
            ], lattice_in, center=(2, 2))
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
                 activation_function: Callable = ncnn.LeakyReLU,
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

        self.pre_downsample = nn.Identity()
        # if lattice_in == ncdl.Lattice('cp'):
        #     print("...down")
        #     self.pre_downsample = nn.Sequential(
        #         ncnn.LatticeUnwrap(),
        #         nn.MaxPool2d(kernel_size=2, stride=1, ceil_mode=True),
        #         ncnn.LatticeWrap()
        #     )


        self.downsample = ncnn.LatticeDownsample(lattice_in, np.array(
            resample_matrix[resample_rate],
        dtype='int'))

    @property
    def lattice_out(self):
        return self.downsample.down_lattice

    def forward(self, lt: ncdl.LatticeTensor):
        lt = self.block(lt)
        return lt, self.pre_downsample(self.downsample(lt))


class UpBlock(nn.Module):
    def __init__(self,
                 lattice_in: ncdl.Lattice,
                 channels_in: int,
                 channels_out: int,
                 skip_input: Optional[int] = 0,
                 block_type: str = 'basic',
                 normalization: Optional[str] = None,
                 skip_first: bool = False,
                 activation_function: Callable = ncnn.LeakyReLU,
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
            print("??")
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
                 activation_function: Callable = ncnn.LeakyReLU):
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
                channels_out if skip else 0,
                block_type,
                norm,
                False,
                activation,
                down_sample,

            )

    def forward(self, lt: ncdl.LatticeTensor, layer=0, iter=0):
        # if lt.parent.coset_count
        # print([lt.coset(_).shape for _ in range(lt.parent.coset_count)])

        if lt.parent == ncdl.Lattice('cp') and (iter + 1) % 10000 == 0:
            shape = lt.coset(0).shape
            for _ in range(shape[-3]):
                path = os.path.join("debug_cp", f"layer_{layer}")
                pathlib.Path(path).mkdir(exist_ok=True, parents=True)
                with torch.no_grad():
                    coset = lt.coset(0)[0:1, _:_+1, :, : ]
                    coset = (coset - coset.min())/(coset.max() - coset.min())
                    coset = torch.nn.functional.interpolate(coset, 256, mode='bilinear')[0]

                    torchvision.transforms.functional.to_pil_image(coset).save(os.path.join(path, f"cp_{layer}_{shape[-2]}x{shape[-1]}_{_}.png"))
                # print(path)

        # print(lt.coset(0).shape)
        if self.down_block:
            lt, lt_down = self.down_block(lt)
        else:
            # print(lt.coset(0).sum())
            lt, lt_down = None, lt

        if isinstance(self.inner_block, InnerRecursiveNetwork):
            lt_down = self.inner_block(lt_down, layer=layer + 1, iter=iter)
        else:
            lt_down = self.inner_block(lt_down)

        if self.up_block:
            return self.up_block(lt_down, lt)
        return lt_down


class Unet(nn.Module):
    def __init__(self, channels_in, channels_out, config, residual: bool = False):
        super().__init__()
        lattice_cp = ncdl.Lattice('cp')
        self.lattice = lattice_cp

        normalization = 'gn' if residual else None
        block_type = 'unet' if not residual else 'residual'

        if config == 'ae_cp_std':
            self.inner_network = InnerRecursiveNetwork(
                lattice_cp,
                [(channels_in, 64, 4,   64, False, None, ncnn.LeakyReLU, True, 'basic'),
                 (64,         128, 4, None, False, None, ncnn.LeakyReLU, False, 'basic'),
                 (128,        256, 4, None, False, None, ncnn.LeakyReLU, False, 'basic'),
                 (256,        512, 4, None, False, None, ncnn.LeakyReLU, False, 'basic'),
                 (512,        1024, 4, None, False, None, ncnn.LeakyReLU, False, 'basic')],
                inner_channels=1024,
                middle_channels=1024
            )
            self.epilogue = nn.Conv2d(64, channels_out, kernel_size=1, padding=0)
        elif config == 'ae_cp_dbl':
            print(
                'cp_dob'
            )
            self.inner_network = InnerRecursiveNetwork(
                lattice_cp,
                [(channels_in, 64, 4,   64, False, None, ncnn.LeakyReLU, True, 'basic'),
                 (64,          64, 1, None, False, None, ncnn.LeakyReLU, False, 'basic'),

                 (64,         128, 4, None, False, None, ncnn.LeakyReLU, False, 'basic'),
                 (128,        128, 1, None, False, None, ncnn.LeakyReLU, False, 'basic'),

                 (128,        256, 4, None, False, None, ncnn.LeakyReLU, False, 'basic'),
                 (256,        256, 1, None, False, None, ncnn.LeakyReLU, False, 'basic'),

                 (256,        512, 4, None, False, None, ncnn.LeakyReLU, False, 'basic'),
                 (512,        512, 1, None, False, None, ncnn.LeakyReLU, False, 'basic'),

                 (512, 512, 4, None, False, None, ncnn.LeakyReLU, False, 'basic'),
                 (512, 1024, 1, None, False, None, ncnn.LeakyReLU, False, 'basic'),

                 (1024, 1024, 4, None, False, None, ncnn.LeakyReLU, False, 'basic'),
                 (1024, 1024, 1, None, False, None, ncnn.LeakyReLU, False, 'basic'),
                 (1024, 1024, 4, None, False, None, ncnn.LeakyReLU, False, 'basic'),
                 (1024, 1024, 1, None, False, None, ncnn.LeakyReLU, False, 'basic'),
                 (1024, 1024, 4, None, False, None, ncnn.LeakyReLU, False, 'basic'),
                 (1024, 1024, 1, None, False, None, ncnn.LeakyReLU, False, 'basic'),

                 ],
                inner_channels=1024,
                middle_channels=1024
            )
            self.epilogue = nn.Conv2d(64, channels_out, kernel_size=1, padding=0)
        elif config == 'ae_qc_dbl':
            self.inner_network = InnerRecursiveNetwork(
                lattice_cp,
                [(channels_in, 64, 2, 64, False, None, ncnn.LeakyReLU, True, 'basic'),
                 (64, 64, 2, None, False, None, ncnn.LeakyReLU, False, 'basic'),

                 (64, 128, 2, None, False, None, ncnn.LeakyReLU, False, 'basic'),
                 (128, 128, 2, None, False, None, ncnn.LeakyReLU, False, 'basic'),

                 (128, 256, 2, None, False, None, ncnn.LeakyReLU, False, 'basic'),
                 (256, 256, 2, None, False, None, ncnn.LeakyReLU, False, 'basic'),

                 (256, 512, 2, None, False, None, ncnn.LeakyReLU, False, 'basic'),
                 (512, 512, 2, None, False, None, ncnn.LeakyReLU, False, 'basic'),

                 (512, 512, 2, None, False, None, ncnn.LeakyReLU, False, 'basic'),
                 (512, 1024, 2, None, False, None, ncnn.LeakyReLU, False, 'basic'),

                 (1024, 1024, 2, None, False, None, ncnn.LeakyReLU, False, 'basic'),
                 (1024, 1024, 2, None, False, None, ncnn.LeakyReLU, False, 'basic'),

                 (1024, 1024, 2, None, False, None, ncnn.LeakyReLU, False, 'basic'),
                 (1024, 1024, 2, None, False, None, ncnn.LeakyReLU, False, 'basic'),

                 (1024, 1024, 2, None, False, None, ncnn.LeakyReLU, False, 'basic'),
                 (1024, 1024, 2, None, False, None, ncnn.LeakyReLU, False, 'basic'),
                 ],
                inner_channels=1024,
                middle_channels=1024
            )
            self.epilogue = nn.Conv2d(64, channels_out, kernel_size=1, padding=0)
        elif config == 'ae_qc_grdl':
            self.inner_network = InnerRecursiveNetwork(
                lattice_cp,
                [(channels_in, 64, 2, 64, False, None, ncnn.LeakyReLU, True, 'basic'),
                 (64, 96, 2, None, False, None, ncnn.LeakyReLU, False, 'basic'),

                 (96, 128, 2, None, False, None, ncnn.LeakyReLU, False, 'basic'),
                 (128, 192, 2, None, False, None, ncnn.LeakyReLU, False, 'basic'),

                 (192, 256, 2, None, False, None, ncnn.LeakyReLU, False, 'basic'),
                 (256, 384, 2, None, False, None, ncnn.LeakyReLU, False, 'basic'),

                 (384, 512, 2, None, False, None, ncnn.LeakyReLU, False, 'basic'),
                 (512, 512, 2, None, False, None, ncnn.LeakyReLU, False, 'basic')],
                inner_channels=512,
                middle_channels=1024
            )
            self.epilogue = nn.Conv2d(64, channels_out, kernel_size=1, padding=0)
        elif config == 'ae_cp_grdl':
            self.inner_network = InnerRecursiveNetwork(
                lattice_cp,
                [(channels_in, 64, 4, 64, False, None, ncnn.LeakyReLU, True, 'basic'),
                 (64, 96, 1, None, False, None, ncnn.LeakyReLU, False, 'basic'),

                 (96, 128, 4, None, False, None, ncnn.LeakyReLU, False, 'basic'),
                 (128, 192, 1, None, False, None, ncnn.LeakyReLU, False, 'basic'),

                 (192, 256, 4, None, False, None, ncnn.LeakyReLU, False, 'basic'),
                 (256, 384, 1, None, False, None, ncnn.LeakyReLU, False, 'basic'),

                 (384, 512, 4, None, False, None, ncnn.LeakyReLU, False, 'basic'),
                 (512, 512, 1, None, False, None, ncnn.LeakyReLU, False, 'basic')],
                inner_channels=512,
                middle_channels=1024
            )
            self.epilogue = nn.Conv2d(64, channels_out, kernel_size=1, padding=0)

        elif config == 'unet_cp_std':
            self.inner_network = InnerRecursiveNetwork(
                lattice_cp,
                [(channels_in, 64, 4,   64, True, normalization, ncnn.LeakyReLU, True, block_type),
                 (64,         128, 4, None, True, normalization, ncnn.LeakyReLU, False, block_type),
                 (128,        256, 4, None, True, normalization, ncnn.LeakyReLU, False, block_type),
                 (256,        512, 4, None, True, normalization, ncnn.LeakyReLU, False, block_type)],
                inner_channels=512,
                middle_channels=1024
            )
            self.epilogue = nn.Conv2d(64, channels_out, kernel_size=1, padding=0)
        elif config == 'unet_cp_dbl':
            self.inner_network = InnerRecursiveNetwork(
                lattice_cp,
                [(channels_in, 64, 4,   64, False, normalization, ncnn.LeakyReLU, True, block_type),
                 (64,          64, 1, None, True, normalization, ncnn.LeakyReLU, False, block_type),

                 (64,         128, 4, None, False, normalization, ncnn.LeakyReLU, False, block_type),
                 (128,        128, 1, None, True, normalization, ncnn.LeakyReLU, False, block_type),

                 (128,        256, 4, None, False, normalization, ncnn.LeakyReLU, False, block_type),
                 (256,        256, 1, None, True, normalization, ncnn.LeakyReLU, False, block_type),

                 (256,        512, 4, None, False, normalization, ncnn.LeakyReLU, False, block_type),
                 (512,        512, 1, None, True, normalization, ncnn.LeakyReLU, False, block_type)
                 ],
                inner_channels=512,
                middle_channels=1024
            )
            self.epilogue = nn.Conv2d(64, channels_out, kernel_size=1, padding=0)
        elif config == 'unet_cp_dbl_skip':
            self.inner_network = InnerRecursiveNetwork(
                lattice_cp,
                [(channels_in, 64, 4,   64, True, normalization, ncnn.LeakyReLU, True, block_type),
                 (64,          64, 1, None, True, normalization, ncnn.LeakyReLU, False, block_type),

                 (64,         128, 4, None, True, normalization, ncnn.LeakyReLU, False, block_type),
                 (128,        128, 1, None, True, normalization, ncnn.LeakyReLU, False, block_type),

                 (128,        256, 4, None, True, normalization, ncnn.LeakyReLU, False, block_type),
                 (256,        256, 1, None, True, normalization, ncnn.LeakyReLU, False, block_type),

                 (256,        512, 4, None, True, normalization, ncnn.LeakyReLU, False, block_type),
                 (512,        512, 1, None, True, normalization, ncnn.LeakyReLU, False, block_type)
                 ],
                inner_channels=512,
                middle_channels=1024
            )
            self.epilogue = nn.Conv2d(64, channels_out, kernel_size=1, padding=0)
        elif config == 'unet_qc_dbl':
            self.inner_network = InnerRecursiveNetwork(
                lattice_cp,
                [(channels_in, 64, 2, 64, True, normalization, ncnn.LeakyReLU, True, block_type),
                 (64, 64, 2, None, True, normalization, ncnn.LeakyReLU, False, block_type),

                 (64, 128, 2, None, True, normalization, ncnn.LeakyReLU, False, block_type),
                 (128, 128, 2, None, True, normalization, ncnn.LeakyReLU, False, block_type),

                 (128, 256, 2, None, True, normalization, ncnn.LeakyReLU, False, block_type),
                 (256, 256, 2, None, True, normalization, ncnn.LeakyReLU, False, block_type),

                 (256, 512, 2, None, True, normalization, ncnn.LeakyReLU, False, block_type),
                 (512, 512, 2, None, True, normalization, ncnn.LeakyReLU, False, block_type)],
                inner_channels=512,
                middle_channels=1024
            )
            self.epilogue = nn.Conv2d(64, channels_out, kernel_size=1, padding=0)
        elif config == 'unet_cp_grdl':
            self.inner_network = InnerRecursiveNetwork(
                lattice_cp,
                [(channels_in, 64, 4,   64, False, normalization, ncnn.LeakyReLU, True, block_type),
                 (64,          96, 1, None, True, normalization, ncnn.LeakyReLU, False, block_type),

                 (96,         128, 4, None, False, normalization, ncnn.LeakyReLU, False, block_type),
                 (128,        192, 1, None, True, normalization, ncnn.LeakyReLU, False, block_type),

                 (192,        256, 4, None, False, normalization, ncnn.LeakyReLU, False, block_type),
                 (256,        384, 1, None, True, normalization, ncnn.LeakyReLU, False, block_type),

                 (384,        512, 4, None, False, normalization, ncnn.LeakyReLU, False, block_type),
                 (512,        512, 1, None, True, normalization, ncnn.LeakyReLU, False, block_type)
                 ],
                inner_channels=512,
                middle_channels=1024
            )
            self.epilogue = nn.Conv2d(64, channels_out, kernel_size=1, padding=0)

        elif config == 'unet_qc_grdl':
            self.inner_network = InnerRecursiveNetwork(
                lattice_cp,
                [(channels_in, 64, 2, 64, False, normalization, ncnn.LeakyReLU, True, block_type),
                 (64, 96, 2, None, True, normalization, ncnn.LeakyReLU, False, block_type),

                 (96, 128, 2, None, False, normalization, ncnn.LeakyReLU, False, block_type),
                 (128, 192, 2, None, True, normalization, ncnn.LeakyReLU, False, block_type),

                 (192, 256, 2, None, False, normalization, ncnn.LeakyReLU, False, block_type),
                 (256, 384, 2, None, True, normalization, ncnn.LeakyReLU, False, block_type),

                 (384, 512, 2, None, False, normalization, ncnn.LeakyReLU, False, block_type),
                 (512, 512, 2, None, False, normalization, ncnn.LeakyReLU, False, block_type)],
                inner_channels=512,
                middle_channels=1024
            )
            self.epilogue = nn.Conv2d(64, channels_out, kernel_size=1, padding=0)
        else:
            raise NotImplementedError(f'Configuration {config} not implemented')

    def forward(self, x, iter=0):
        lt = self.lattice(x)
        lt = self.inner_network(lt, iter=iter)
        return nn.functional.sigmoid(self.epilogue(lt.coset(0)))