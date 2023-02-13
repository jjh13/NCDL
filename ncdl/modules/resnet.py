import torch
import torch.nn as nn

from ncdl.lattice import Lattice
from ncdl.nn import *
# from ncdl.nn.modules import ReLU
# from ncdl.nn.modules import LatticeConvolution
# from ncdl.nn.modules import LatticeBatchNorm
# from ncdl.nn.modules import LatticeDownsample
# from ncdl.nn.modules import LatticeWrap, LatticeUnwrap
# from ncdl.nn.modules import LatticePad
from ncdl.util.stencil import Stencil



class QCCPResidualBlock(nn.Module):
    def __init__(self,
                 lattice_in,
                 in_channels,
                 out_channels,
                 downsample=1):
        super(QCCPResidualBlock, self).__init__()

        # Define the stencils that we use for the filters on different lattices
        cp_stencil = [(0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2), (2, 0), (2, 1), (2, 2)]
        cp_center = (1,1)
        cp_s = Stencil(cp_stencil, Lattice("cp"), center=cp_center)

        # qc_stencil = [(1, 1), (2, 2), (3, 1), (1, 3), (3, 3), (0, 2), (2, 0), (0, 4), (4, 0)]
        qc_stencil = [(0, 0), (2, 0), (0, 2), (2, 2), (1, 1)]
        qc_center = (1, 1)
        qc_s = Stencil(qc_stencil, Lattice("qc"), center=qc_center)
        lattice_out = None

        # Choose the target lattice
        if downsample == 1:
            downsample = nn.Identity()
            self.project = nn.Sequential(
                LatticeConvolution(lattice_in, in_channels, out_channels, Stencil([(0,0)], lattice_in, center=(0,0))),
                LatticeBatchNorm(lattice_in, out_channels)
            )
            lattice_out = lattice_in

        elif downsample == 2:
            downsample = LatticeDownsample(lattice_in, torch.IntTensor([[1, 1], [1, -1]]))

            self.project = nn.Sequential(
                downsample,
                LatticeConvolution(downsample.down_lattice, in_channels, out_channels, Stencil([(0,0)], downsample.down_lattice, center=(0,0))),
                LatticeBatchNorm(downsample.down_lattice, out_channels)
            )
            lattice_out = downsample.down_lattice

        elif downsample == 4:
            downsample = LatticeDownsample(lattice_in, torch.IntTensor(torch.IntTensor([[2, 0], [0, 2]])))
            self.project = nn.Sequential(
                downsample,
                LatticeConvolution(lattice_in, in_channels, out_channels, Stencil([(0,0)], downsample.down_lattice, center=(0,0))),
                LatticeBatchNorm(downsample.down_lattice, out_channels)
            )
            lattice_out = downsample.down_lattice
        self.lattice_out = lattice_out


        if lattice_in == Lattice("cp"):
            self.conv1 = nn.Sequential(
                LatticePad(lattice_in, cp_s),
                LatticeConvolution(lattice_in, in_channels, out_channels, cp_s),
                LatticeBatchNorm(lattice_in, out_channels),
                downsample,
                ReLU(inplace=True))


        elif lattice_in == Lattice("qc"):
            self.conv1 = nn.Sequential(
                LatticePad(lattice_in, qc_s),
                LatticeConvolution(lattice_in, in_channels, out_channels, qc_s),
                LatticeBatchNorm(lattice_in, out_channels),
                downsample,
                ReLU(inplace=True))
        else:
            raise ValueError("Invalid lattice_in! Should be quincunx or cartesian planar")


        if lattice_out == Lattice("cp"):
            self.conv2 = nn.Sequential(
                LatticePad(lattice_out, cp_s),
                LatticeConvolution(lattice_out, out_channels, out_channels, cp_s),
                LatticeBatchNorm(lattice_out, out_channels),
            )
        elif lattice_out == Lattice("qc"):
            self.conv2 = nn.Sequential(
                LatticePad(lattice_out, qc_s),
                LatticeConvolution(lattice_out, out_channels, out_channels, qc_s),
                LatticeBatchNorm(lattice_out, out_channels)
            )

        self.relu_out = ReLU(inplace=True)

    @property
    def down_lattice(self):
        return self.lattice_out

    def forward(self, x):
        residual = self.project(x)
        out = self.conv1(x)
        out = self.conv2(out)
        return self.relu_out(out + residual)

class GlobalAveragePool2d(nn.Module):
    def __init__(self):
        super(GlobalAveragePool2d, self).__init__()

    def forward(self, x):
        return nn.avg_pool2d(x, x.size()[2:])


class Resnet18(nn.Module):
    def __init__(self, numclasses=1000,variant=2):
        super(Resnet18, self).__init__()

        self.preamble = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=(7, 7), stride=(2, 2), padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            LatticeWrap()
        )

        if variant==0:
            lattice, conv1 = Resnet18._make_layer(64, 64, 3, 1, Lattice("cp"))
            lattice, conv2 = Resnet18._make_layer(64, 128, 4, 4, lattice)
            lattice, conv3 = Resnet18._make_layer(128, 256, 6, 4, lattice)
            lattice, conv4 = Resnet18._make_layer(256, 512, 3, 4, lattice)

            self.res_conv = nn.Sequential(
                conv1,
                conv2,
                conv3,
                conv4
            )
        elif variant==1:
            lattice, conv1 = Resnet18._make_layer(64, 64, 3*2, 1, Lattice("cp"))
            lattice, conv2 = Resnet18._make_layer(64, 128, 4*2, 4, lattice)
            lattice, conv3 = Resnet18._make_layer(128, 256, 6*2, 4, lattice)
            lattice, conv4 = Resnet18._make_layer(256, 512, 3*2, 4, lattice)

            self.res_conv = nn.Sequential(
                conv1,
                conv2,
                conv3,
                conv4
            )
        elif variant == 2:
            lattice, conv1  = Resnet18._make_layer(64, 64, 3, 1, Lattice("cp"))
            lattice, conv1a = Resnet18._make_layer(64, 64, 3, 2, lattice)

            lattice, conv2 = Resnet18._make_layer(64, 128, 4, 2, lattice)
            lattice, conv2a = Resnet18._make_layer(128, 128, 4, 2, lattice)

            lattice, conv3 = Resnet18._make_layer(128, 256, 6, 2, lattice)
            lattice, conv3a = Resnet18._make_layer(256, 256, 6, 2, lattice)

            lattice, conv4 = Resnet18._make_layer(256, 512, 3, 2, lattice)
            lattice, conv4a = Resnet18._make_layer(512, 512, 3, 1, lattice)

            self.res_conv = nn.Sequential(
                conv1, conv1a,
                conv2, conv2a,
                conv3, conv3a,
                conv4, conv4a
            )
        self.postamble = nn.Sequential(
            LatticeUnwrap(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(512, numclasses)
        )

    @classmethod
    def _make_layer(cls, channels_in, channels_out, layers, down_factor, lattice_in):

        blocks = []
        for layer_idx in range(layers):

            if layer_idx == 0:
                blk = QCCPResidualBlock(lattice_in, channels_in, channels_out, down_factor)
                lattice_in = blk.down_lattice if down_factor != 1 else lattice_in
            else:
                blk = QCCPResidualBlock(lattice_in, channels_out, channels_out, 1)
            blocks += [blk]

        return lattice_in, nn.Sequential(*blocks)


    def forward(self, x):
        x = self.preamble(x)
        x = self.res_conv(x)
        return self.postamble(x)
