import torch
import torch.nn as nn

from ncdl.lattice import Lattice
from ncdl.nn import *
from ncdl.util.stencil import Stencil


class NormalResidualBlock(nn.Module):
    def __init__(self,
                 lattice_in,
                 in_channels,
                 out_channels,
                 downsample=1):
        super(NormalResidualBlock, self).__init__()
        pass


class QCCPResidualBlock(nn.Module):
    def __init__(self,
                 lattice_in,
                 in_channels,
                 out_channels,
                 downsample=1):
        super(QCCPResidualBlock, self).__init__()

        # Define the stencils that we use for the filters on different lattices
        cp_stencil = [(0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2), (2, 0), (2, 1), (2, 2)]
        cp_center = (1, 1)
        cp_s = Stencil(cp_stencil, Lattice("cp"), center=cp_center)

        # qc_stencil = [(1, 1), (2, 2), (3, 1), (1, 3), (3, 3), (0, 2), (2, 0), (0, 4), (4, 0)]
        qc_stencil = [(0, 0), (2, 0), (0, 2), (2, 2), (1, 1)]
        qc_center = (1, 1)
        qc_s = Stencil(qc_stencil, Lattice("qc"), center=qc_center)
        lattice_out = None

        # Choose the target lattice
        if downsample == 1:
            print("ds1")
            downsample = nn.Identity()
            if in_channels != out_channels:
                self.project = nn.Sequential(
                    # LatticeConvolution(
                    #     lattice_in,
                    #     in_channels,
                    #     out_channels,
                    #     Stencil([(0,0)], lattice_in, center=(0,0)),
                    #     bias=False),
                        LatticeUnwrap(),
                        nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0, bias=False),
                        nn.BatchNorm2d(out_channels),
                        LatticeWrap()
                # LatticeBatchNorm(lattice_in, out_channels)
                )
            else:
                self.project = nn.Identity()
            lattice_out = lattice_in
            dsint = 1

        elif downsample == 2:
            print("ds2")

            downsample = LatticeDownsample(lattice_in, torch.IntTensor([[1, 1], [1, -1]]))

            self.project = nn.Sequential(
                downsample,
                LatticeConvolution(downsample.down_lattice, in_channels, out_channels, Stencil([(0,0)], downsample.down_lattice, center=(0,0)), bias=False),
                LatticeBatchNorm(downsample.down_lattice, out_channels)
            )
            lattice_out = downsample.down_lattice
            dsint = 2
        elif downsample == 4:
            print("ds4")

            downsample = LatticeDownsample(lattice_in, torch.IntTensor([[2, 0], [0, 2]]))
            self.project = nn.Sequential(
                # downsample,
                # LatticeConvolution(
                #     lattice_in,
                #     in_channels,
                #     out_channels,
                #     Stencil([(0,0)], downsample.down_lattice, center=(0,0)),
                #     bias=False
                # ),
                LatticeUnwrap(),
                nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0, bias=False, stride=2),
                nn.BatchNorm2d(out_channels),
                LatticeWrap()
            # LatticeBatchNorm(downsample.down_lattice, out_channels)
            )
            lattice_out = downsample.down_lattice
            dsint = 4
        self.lattice_out = lattice_out


        if lattice_in == Lattice("cp"):
            self.conv1 = nn.Sequential(
                LatticeUnwrap(),
                nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False, stride=2 if dsint == 4 else 1),
                nn.BatchNorm2d(out_channels),
                LatticeWrap(),
                # LatticePad(lattice_in, cp_s),
                # LatticeConvolution(lattice_in, in_channels, out_channels, cp_s, bias=False),
                # LatticeBatchNorm(lattice_in, out_channels),
                # downsample,
                ReLU(inplace=False))


        elif lattice_in == Lattice("qc"):
            self.conv1 = nn.Sequential(
                # LatticeUnwrap(),
                nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False, stride=1),
                nn.BatchNorm2d(out_channels),
                # LatticePad(lattice_in, qc_s),
                # LatticeConvolution(lattice_in, in_channels, out_channels, qc_s, bias=False),
                # LatticeBatchNorm(lattice_in, out_channels),
                # downsample,
                nn.ReLU(inplace=False))
        else:
            raise ValueError("Invalid lattice_in! Should be quincunx or cartesian planar")


        if lattice_out == Lattice("cp"):
            self.conv2 = nn.Sequential(
                # LatticePad(lattice_out, cp_s),
                # LatticeConvolution(lattice_out, out_channels, out_channels, cp_s, bias=False),
                LatticeUnwrap(),
                nn.ConstantPad2d((1,1,1,1), 0.0),
                nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=0, bias=False),
                nn.BatchNorm2d(out_channels),
                LatticeWrap()
                # LatticeBatchNorm(lattice_out, out_channels),
            )
        elif lattice_out == Lattice("qc"):
            self.conv2 = nn.Sequential(
                LatticePad(lattice_out, qc_s),
                LatticeConvolution(lattice_out, out_channels, out_channels, qc_s, bias=False),
                LatticeBatchNorm(lattice_out, out_channels)
            )

        self.relu_out = ReLU(inplace=False)

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
        return nn.functional.avg_pool2d(x, x.size()[2:])


class Resnet18(nn.Module):
    def __init__(self, numclasses=1000,variant=2):
        super(Resnet18, self).__init__()

        self.preamble = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=(7, 7), stride=(2, 2), padding=3, bias=False),
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

        for m in self.modules():
            if isinstance(m, LatticeConvolution):
                print("?")
                pre = m.get_convolution_weights(0).clone()
                nn.init.kaiming_normal_(m.get_convolution_weights(0), mode="fan_out", nonlinearity="relu")
                print("!", m.get_convolution_weights(0)-pre)

            elif isinstance(m, nn.Conv2d):
                pass
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (LatticeBatchNorm, LatticeGroupNorm, nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

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
