import unittest
import torch
from ncdl.lattice import Lattice
import ncdl.nn as ncnn
from ncdl.nn import *
from ncdl.modules.resnet import QCCPResidualBlock, Resnet18
from ncdl.modules.autoencoder import ConvBlock, DownBlock, UpBlock, InnerRecursiveNetwork, Unet, DoubleConv,Up, Down

from torchvision.models.resnet import BasicBlock, conv3x3, conv1x1, resnet50
from ncdl.util.stencil import Stencil
from torch import nn
import numpy as np

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

class LatticeConstruction(unittest.TestCase):
    def setUp(self):
        self.devices = [torch.device('cpu')]
        if torch.cuda.is_available():
            self.devices += [torch.device('cuda:0')]
        elif torch.backends.mps.is_available():
            self.devices += [torch.device('mps:0')]

    def test_conv(self):
        lattice_cp = Lattice("cp")
        lattice_qc = Lattice("qc")

        for block_type in ['basic', 'residual', 'unet']:
            print(block_type)
            cbcp = ConvBlock(lattice_cp, 128, 256, block_type, 'bn')
            cbqc = ConvBlock(lattice_qc, 128, 256, block_type, 'bn')

            ltcp = lattice_cp(torch.rand(1, 128, 64, 64))
            ltqc = lattice_qc(torch.rand(1, 128, 64, 64), torch.rand(1, 128, 64, 64))

            cbcp(ltcp)
            cbqc(ltqc)

    def test_double_conv(self):
        lattice = Lattice("cp")
        dc = nn.Sequential(
            DoubleConv(lattice, 16, 16, 8),
            Down(lattice, 16, 16, 4)
            # DoubleConv(lattice, 16, 16, 8),
            # Up(lattice, 16, 16, 4),
        )
        # up = Up(lattice, 16, 16, 4)

        rg = torch.rand(1, 16, 128, 128, requires_grad=True)

        output = dc(lattice(rg)) #, lattice(rg[:, :8,:,:]))
        loss = (output.coset(0).sum()- 1)**2
        loss.backward()

        print(rg.grad)



    def test_down(self):
        lattice_cp = Lattice("cp")
        lattice_qc = Lattice("qc")

        for block_type in ['basic', 'residual', 'unet']:
            print(block_type)
            cbcp = DownBlock(lattice_cp, 128, 256, block_type, 'bn')
            cbqc = DownBlock(lattice_qc, 128, 256, block_type, 'bn')

            ltcp = lattice_cp(torch.rand(1, 128, 64, 64))
            ltqc = lattice_qc(torch.rand(1, 128, 64, 64), torch.rand(1, 128, 64, 64))

            skip_cp, ltoqc = cbcp(ltcp)
            skip_qc, ltocp = cbqc(ltqc)

            self.assertEqual(ltocp.parent, lattice_cp)
            self.assertEqual(skip_qc.parent, lattice_qc)

            self.assertEqual(ltoqc.parent, lattice_qc)
            self.assertEqual(skip_cp.parent, lattice_cp)


            up_cp = UpBlock(lattice_cp, 256, 128, block_type=block_type, normalization='bn')
            ltcp_ = up_cp(ltocp, skip_qc)
            up_cp = UpBlock(lattice_qc, 256, 128, block_type=block_type, normalization='bn')
            ltqc_ = up_cp(ltoqc, skip_cp)


            up_cp = UpBlock(lattice_cp, 256, 128, skip_input=256, block_type=block_type, normalization='bn')
            ltcp_ = up_cp(ltocp, skip_qc)
            up_cp = UpBlock(lattice_qc, 256, 128, skip_input=256, block_type=block_type, normalization='bn')
            ltqc_ = up_cp(ltoqc, skip_cp)

    def test_recursive_module(self):
        lattice_cp = Lattice("cp")
        netowrk = InnerRecursiveNetwork(
            lattice_cp,
            [(1, 64, 4, True, 'bn', ncnn.ReLU, False, 'basic'),
             (64, 128, 2, True, 'bn', ncnn.ReLU, False, 'basic'),
             (128, 256, 2, True, 'bn', ncnn.ReLU, False, 'basic'),
             (256, 512, 4, True, 'bn', ncnn.ReLU, False, 'basic')],
            inner_channels=512,
            middle_channels=1024
        )

        lt = lattice_cp(torch.rand(1, 1, 256, 256))
        lto = netowrk(lt)
        print(lto.coset(0).shape)

    def test_unet_autoencoder_structs(self):

        for config in ['ae_cp_std', 'ae_cp_dbl', 'ae_qc_dbl', 'ae_cp_dbl', 'ae_cp_grdl', 'ae_qc_grdl']:
            network = Unet(3, 1, config)
            network(torch.rand(1, 3, 256, 256))


    def test_unet_structs(self):
        for config in ['unet_cp_std', 'unet_cp_dbl', 'unet_qc_dbl', 'unet_cp_grdl', 'unet_qc_grdl']:
            network = Unet(3, 1, config)
            network(torch.rand(1, 3, 256, 256))


    def test_unet_structs_res(self):
        for config in ['unet_cp_std', 'unet_cp_dbl', 'unet_qc_dbl', 'unet_cp_grdl', 'unet_qc_grdl']:
            network = Unet(3, 1, config, residual=True)
            network(torch.rand(1, 3, 256, 256))

    def test_ae_dissappearance(self):
        from ncdl.modules.autoencoder import Unet