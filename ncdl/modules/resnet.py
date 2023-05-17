import torch
import torch.nn as nn

from ncdl.lattice import Lattice, LatticeTensor
from ncdl.nn.functional.convolution import lattice_conv
from ncdl.nn import *
from ncdl.util.stencil import Stencil
from ncdl.modules.diet import DietQCModule

class NormalResidualBlock(nn.Module):
    def __init__(self,
                 lattice_in,
                 in_channels,
                 out_channels,
                 downsample=1):
        super(NormalResidualBlock, self).__init__()
        pass


class QCIdentityConv(nn.Module):
    def __init__(self,
                 lattice: Lattice,
                 channels_in: int,
                 channels_out: int,
                 stencil: "Stencil",
                 groups: int = 1,
                 bias: bool = True):
        """

        :param lattice:
        :param channels_in:
        """
        super().__init__()

        # Typecheck all inputs
        type_fails = []
        if not isinstance(lattice, Lattice):
            type_fails += [('lattice', 'Lattice', type(lattice))]

        if not isinstance(channels_in, int):
            type_fails += [('channels_in', 'int', type(channels_in))]

        if not isinstance(channels_out, int):
            type_fails += [('channels_out', 'int', type(channels_out))]

        if not isinstance(groups, int):
            type_fails += [('groups', 'int', type(groups))]

        if not isinstance(bias, bool):
            type_fails += [('bias', 'bool', type(bias))]

        if len(type_fails):
            print(type_fails)
            raise TypeError(
                f"LatticeConvolution instantiated with invalid or malformed parameters: " + ", ".join([
                    f"'{param}' should be of type '{expected}', not '{obs}'" for param, expected, obs in type_fails
                ])
            )

        if channels_in % groups != 0:
            raise ValueError("Input channel number must be divisible by 'groups'")

        self._stencil = stencil
        self._lattice = lattice
        self._channels_out = channels_out
        self._channels_in = channels_in
        # TODO: Add groups
        self._groups = groups
        self._bias = None

        """
        Setup the weights for each coset's stencil. Note that the 
        """
        coset_stencils = stencil.coset_decompose(packed_output=True)
        for coset_id, stencil in enumerate(coset_stencils):
            if len(stencil) == 0:
                continue
            # Regardless of the stencil type, we zero out the stencils here
            # weights = nn.Parameter(torch.empty(channels_out, channels_in, len(stencil)))
            if coset_id == 0:
                weights = torch.zeros(channels_out, channels_in, len(stencil))
            if coset_id == 1:
                assert channels_in == channels_out
                assert len(stencil) == 1
                weights = torch.zeros(channels_out, channels_in, len(stencil))
                for _ in range(channels_out):
                    weights[_, _, 0] = 1




            if not self._stencil.is_coset_square(coset_id):
                stencil_index, _ = self._stencil.weight_index(coset_id)
                self.register_buffer(f"stencil_index_{coset_id}", torch.IntTensor(stencil_index))
            self.register_buffer(f"weight_{coset_id}", weights)
        if bias is True:
            raise TypeError()
        # self.reset_parameters()

    def reset_parameters(self) -> None:
        return
        k = len(self._stencil.stencil) * self._channels_out

        for idx in range(self._lattice.coset_count):
            if hasattr(self, f"weight_{idx}"):
                w = self.get_parameter(f"weight_{idx}")
                init.uniform_(self.get_parameter(f"weight_{idx}"), -1/math.sqrt(k), 1/math.sqrt(k))

        if self._bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(w)
            if fan_in != 0:
                bound = 1 / math.sqrt(fan_in)
                init.uniform_(self._bias, -bound, bound)

    def get_convolution_weights(self, coset_id):
        """
        Returns the /square/ convolution weights for this coset. Internally, conv weights are stored in params of
        shape (c,k,len(stencil), this needs to be unpacked into a 4-d tensor. The Stencil class handles this, padding
        in extra zeros if neccesary.
        """

        if not hasattr(self, f"weight_{coset_id}"):
            return None

        weights = self.get_buffer(f"weight_{coset_id}")
        index = None
        if not self._stencil.is_coset_square(coset_id):
            index = self.get_buffer(f"stencil_index_{coset_id}")
        return self._stencil.unpack_weights(coset_id, weights, index)

    def forward(self, lt: LatticeTensor) -> LatticeTensor:
        if lt.parent != self._lattice:
            raise ValueError("Bad ")

        weights = [self.get_convolution_weights(i) for i in range(self._lattice.coset_count)]
        return lattice_conv(lt, self._lattice, self._stencil, weights, self._bias, self._groups)



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

        # qc_stencil = [(1, 1), (2, 2), (3, 1), (1, 3), (3, 3), (0, 2), (2, 0), (2, 4), (4, 2)]
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
                    LatticeConvolution(
                        lattice_in,
                        in_channels,
                        out_channels,
                        Stencil([(0,0)], lattice_in, center=(0,0)),
                        bias=False),
                        # LatticeUnwrap(),
                        # nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0, bias=False),
                        # nn.BatchNorm2d(out_channels),
                        # LatticeWrap()
                LatticeBatchNorm(lattice_in, out_channels)
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

            # downsample = LatticeDownsample(lattice_in, torch.IntTensor([[2, 0], [0, 2]]))
            downsample1 = LatticeDownsample(lattice_in, torch.IntTensor([[1, 1], [1, -1]]))

            # id = QCIdentityConv(downsample1.down_lattice, in_channels, in_channels, qc_s, 1, bias=False)


            downsample = LatticeDownsample(downsample1.down_lattice, torch.IntTensor([[1, 1], [1, -1]]))
            self.project = nn.Sequential(
                downsample1,
                # LatticePad(downsample1.down_lattice, qc_s),
                # id,
                downsample,
                LatticeConvolution(
                    downsample.down_lattice,
                    in_channels,
                    out_channels,
                    Stencil([(0,0)], downsample.down_lattice, center=(0,0)),
                    bias=False
                ),
                LatticeBatchNorm(downsample.down_lattice, out_channels)
            )
            downsample = nn.Sequential(
                downsample1,
                downsample
            )
            lattice_out = lattice_in
            dsint = 4
        self.lattice_out = lattice_out


        if lattice_in == Lattice("cp"):
            self.conv1 = nn.Sequential(
                LatticePad(lattice_in, cp_s),
                LatticeConvolution(lattice_in, in_channels, out_channels, cp_s, bias=False),
                downsample,
                LatticeBatchNorm(lattice_out, out_channels),
                ReLU(inplace=False))


        elif lattice_in == Lattice("qc"):
            self.conv1 = nn.Sequential(
                DietQCModule(in_channels=in_channels, out_channels=out_channels),
                # LatticePad(lattice_in, qc_s),
                # LatticeConvolution(lattice_in, in_channels, out_channels, qc_s, bias=False),
                LatticeBatchNorm(lattice_in, out_channels),
                downsample,
                ReLU(inplace=False))
        else:
            raise ValueError("Invalid lattice_in! Should be quincunx or cartesian planar")


        if lattice_out == Lattice("cp"):
            self.conv2 = nn.Sequential(
                LatticePad(lattice_out, cp_s),
                LatticeConvolution(lattice_out, out_channels, out_channels, cp_s, bias=False),
                LatticeBatchNorm(lattice_out, out_channels)
            )

        elif lattice_out == Lattice("qc"):
            self.conv2 = nn.Sequential(
                DietQCModule(in_channels=out_channels, out_channels=out_channels),
                # LatticePad(lattice_out, qc_s),
                # LatticeConvolution(lattice_out, out_channels, out_channels, qc_s, bias=False),
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
                pass
                # print("?")
                # pre = m.get_convolution_weights(0).clone()
                # nn.init.kaiming_normal_(m.get_convolution_weights(0), mode="fan_out", nonlinearity="relu")
                # print("!", m.get_convolution_weights(0)-pre)

            elif isinstance(m, nn.Conv2d):
                pass
                # nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
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
