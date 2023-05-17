import unittest
import torch
from ncdl.lattice import Lattice
from ncdl.nn import *
from ncdl.modules.resnet import QCCPResidualBlock, Resnet18

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
        inout_pairs = [
            (64, 64, 1),
            (64, 128, 1),
            (128, 128, 1),
            (128, 256, 1),
            (256, 256, 1),
            (256, 512, 1),
            (512, 512, 1),
            (64, 64, 2),
            (64, 128, 2),
            (128, 128, 2),
            (128, 256, 2),
            (256, 256, 2),
            (256, 512, 2),
            (512, 512, 2),
        ]
        lattice = Lattice("cp")

        cp_stencil = [(0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2), (2, 0), (2, 1), (2, 2)]
        cp_center = (1, 1)
        cp_s = Stencil(cp_stencil, lattice, center=cp_center)

        for in_channels, out_channels, stride in inout_pairs:
            conv_layer = nn.Sequential(
                conv3x3(in_channels, out_channels, stride, 1, 1),
                nn.BatchNorm2d(out_channels)
            )

            ln_layer = nn.Sequential(
                LatticePad(lattice, cp_s),
                LatticeConvolution(lattice, in_channels, out_channels, cp_s, bias=False),
                LatticeDownsample(lattice, np.array([[stride, 0], [0, stride]], dtype='int')),
                LatticeBatchNorm(lattice, out_channels)
            )

            self.assertEqual(
                count_parameters(conv_layer),
                count_parameters(ln_layer)
            )

    def test_basic_block_param_count(self):

        bb = BasicBlock(128, 128, norm_layer=nn.BatchNorm2d, stride=1)
        bqb = QCCPResidualBlock(Lattice('cp'), 128, 128, downsample=1)

        self.assertEqual(count_parameters(bb), count_parameters(bqb))


    def test_basic_block_param_count_s(self):
        ds = nn.Sequential(
            conv1x1(64, 128, 1),
            nn.BatchNorm2d(128),
        )

        bb = BasicBlock(64, 128, norm_layer=nn.BatchNorm2d, stride=1, downsample=ds)
        bqb = QCCPResidualBlock(Lattice('cp'), 64, 128, downsample=1)

        self.assertEqual(count_parameters(bb), count_parameters(bqb))

    def test_basic_block_param_count_ds(self):
        ds = nn.Sequential(
            conv1x1(64, 128, 2),
            nn.BatchNorm2d(128),
        )

        bb = BasicBlock(64, 128, norm_layer=nn.BatchNorm2d, stride=2, downsample=ds)
        bqb = QCCPResidualBlock(Lattice('cp'), 64, 128, downsample=4)

        self.assertEqual(count_parameters(bb), count_parameters(bqb))

    def test_device_consistent_construction_(self):
        """
        Tests that lattices, when constructed with consistent devices, report the same device.
        """
        qc = Lattice("qc")
        for device in self.devices:
            a0 = torch.ones(1, 3, 16, 16, device=device, requires_grad=True)
            a1 = torch.rand(1, 3, 16, 16, device=device, requires_grad=True)

            lta = qc({
                (0, 0): a0,
                (-1, -1): -a1
            })

            # Q = torch.nn.Sequential(ReLU(True), ReLU(True))
            # relu = Q
            # lta = relu(lta)
            # print(lta.coset(1))

            # ds = LatticeDownsample(qc, torch.IntTensor(torch.IntTensor([[1, 1], [1, -1]])))
            # lta = ds(lta)


            resblock = QCCPResidualBlock(qc, 3, 16, downsample=2)
            resblock = resblock.to(device)
            output = resblock(lta)
            output.coset(0).sum().backward()

            pass

    def test_resnet(self):
        """
        Tests that lattices, when constructed with consistent devices, report the same device.
        """
        qc = Lattice("qc")
        rn = resnet50()

        # 9536
        # 225344
        # 1444928
        # 8543296
        # 23508032
        # 25557032
        for device in self.devices:
            a0 = torch.rand(16, 3, 224, 224, device=device)


            rn = Resnet18().to(device)
            output = rn(a0)

            pass

    def test_device_consistent_construction(self):
        """
        Tests that lattices, when constructed with consistent devices, report the same device.
        """
        qc = Lattice("qc")
        for device in self.devices:
            a0 = torch.ones(1, 3, 3, 3, device=device)
            a1 = torch.zeros(1, 3, 3, 3, device=device)

            lta = qc({
                (0, 0): a0,
                (-1, -1): a1
            })

            self.assertEqual(device, lta.device)

    def test_device_movement(self):
        """
        Tests that lattices can be moved between devices
        """
        qc = Lattice("qc")
        for device in self.devices:
            a0 = torch.ones(1, 3, 3, 3)
            a1 = torch.zeros(1, 3, 3, 3)

            lta = qc({
                (0, 0): a0,
                (-1, -1): a1
            })

            # Test that all the offsets are correct
            src_offsets = [lta.coset_vector(idx) for idx in range(lta.parent.coset_count)]
            src_sizes = [lta.coset(idx).shape for idx in range(lta.parent.coset_count)]

            # Move the tensor to the device
            lta = lta.to(device)
            for idx in range(lta.parent.coset_count):
                for src, trg in zip(lta.coset_vector(idx), src_offsets[idx]):
                    self.assertEqual(src.item(), trg.item())
                for src, trg in zip(lta.coset(idx).shape,  src_sizes[idx]):
                    self.assertEqual(src, trg)

            # Check that all the cosets are on their correct devices
            for coset_idx in range(lta.parent.coset_count):
                self.assertEqual(device, lta.coset(coset_idx).device)


    def test_inconsistent_device_failure(self):
        """
        Tests the failure cases when a lattice is constructed with tensors on different devices
        """
        if len(self.devices) <= 1:
            return

        a0 = torch.ones(1, 3, 3, 3, device=self.devices[0])
        a1 = torch.zeros(1, 3, 3, 3, device=self.devices[1])

        qc = Lattice("qc")
        with self.assertRaises(ValueError):
            lta = qc({
                (0, 0): a0,
                (-1, -1): a1
            })

    def test_lattice_bounds(self):
        """
        Tests that lattice bounds are consistent with an expected case or cases
        """
        for device in self.devices:
            qc = Lattice("qc")
            self.assertEqual(qc.dimension, 2)

            a0 = torch.ones(1, 3, 3, 3)
            a1 = torch.zeros(1, 3, 3, 3)

            lta = qc({
                ( 0,  0): a0,
                (-1, -1): a1
            }).to(device)

            bounds = lta.lattice_bounds()
            self.assertEqual(bounds[0], (0, 0))
            self.assertEqual(bounds[1], (0, 2))
            self.assertEqual(bounds[2], (-1, 4))
            self.assertEqual(bounds[3], (-1, 4))

    def test_create_qc_shifted_add(self):
        """
        Tests that operations between shifted (but consistent) lattices succeed
        """
        for device in self.devices:
            qc = Lattice("qc")
            self.assertEqual(qc.dimension, 2)

            a0 = torch.ones(1, 3, 10, 10).to(device)
            a1 = torch.zeros(1, 3, 11, 11).to(device)

            lta = qc({
                ( 0,  0): a0,
                (-1, -1): a1
            })

            b0 = torch.ones(1, 3, 11, 11).to(device)
            b1 = torch.zeros(1, 3, 10, 10).to(device)

            ltb = qc({
                (0,  0): b0,
                (1, 1): b1
            })

            ltc = lta + ltb

            a = ltc.coset(0) - (a0 + b1)
            b = ltc.coset(1) - (a1 + b0)
            self.assertLess(a.abs().sum().item(), 1e-5)
            self.assertLess(b.abs().sum().item(), 1e-5)

    def test_create_qc_lattice_membership(self):
        for device in self.devices:
            qc = Lattice("qc")
            self.assertEqual(qc.dimension, 2)

            a0 = torch.ones(1, 3, 10, 10)
            a1 = torch.zeros(1, 3, 11, 11)

            lta = qc({
                ( 0,  0): a0,
                (-1, -1): a1
            }).to(device)

            self.assertTrue(lta.on_lattice(torch.IntTensor([-1,-1])))
            self.assertTrue(lta.on_lattice(torch.IntTensor([1,1])))
            self.assertTrue(lta.on_lattice(torch.IntTensor([0,2])))
            self.assertFalse(lta.on_lattice(torch.IntTensor([-1,2])))

    def test_create_qc_add(self):
        """
        Similar to the previous test, but without the shift.
        """
        qc = Lattice("qc")
        for device in self.devices:
            self.assertEqual(qc.dimension, 2)

            a0 = torch.rand(1, 3, 10, 10)
            a1 = torch.rand(1, 3, 9, 9)

            b0 = torch.rand(1, 3, 10, 10)
            b1 = torch.rand(1, 3, 9, 9)

            lta = qc(a0, a1).to(device)
            ltb = qc(b0, b1).to(device)

            ltc = lta + ltb

            self.assertEqual(ltc.coset(0).sum(), (a0.to(device)+b0.to(device)).sum())
            self.assertEqual(ltc.coset(1).sum(), (a1.to(device)+b1.to(device)).sum())


    def test_create_qc_mult(self):
        for device in self.devices:
            qc = Lattice("qc")
            self.assertEqual(qc.dimension, 2)

            a0 = torch.rand(1, 3, 10, 10)
            a1 = torch.rand(1, 3, 9, 9)

            b0 = torch.rand(1, 3, 10, 10)
            b1 = torch.rand(1, 3, 9, 9)

            lta = qc(a0, a1).to(device)
            ltb = qc(b0, b1).to(device)

            ltc = lta * ltb

            self.assertEqual(ltc.coset(0).sum(), (a0.to(device)*b0.to(device)).sum())
            self.assertEqual(ltc.coset(1).sum(), (a1.to(device)*b1.to(device)).sum())

    def test_qc_coset_decompose(self):
        qc = Lattice("qc")
        stencil = [
            (0, 0),
            (1, 1),
            (2, 0),
            (0, 2),
            (2, 2)
        ]
        coset_stencil = qc.coset_decompose(stencil)

        for check_point in [(0,0), (1,0), (0, 1), (1, 1)]:
            self.assertIn(check_point, coset_stencil[0])

        for check_point in [(0,0)]:
            self.assertIn(check_point, coset_stencil[1])

    def test_lattice_slice0(self):
        qc = Lattice("qc")
        self.assertEqual(qc.dimension, 2)

        lt = qc(
            torch.rand(1, 3, 10, 10),
            torch.rand(1, 3, 9, 9)
        )

        lts = lt[:, :, 2:-1, 2:-1]
        print(lts)

    def test_bad_lattice_construct(self):
        # Quincunx Lattice
        qc = Lattice("qc")
        self.assertEqual(qc.dimension, 2)

        coset0 = torch.rand(1, 3, 10, 10)
        coset1 = torch.rand(1, 3, 12, 12)

        with self.assertRaises(ValueError):
            lt = qc(coset0, coset1)

        # BCC Lattice
        bcc = Lattice("BCC")
        self.assertEqual(bcc.dimension, 3)

        with self.assertRaises(ValueError):
            coset0 = torch.rand(1, 3, 10, 10, 10)
            coset1 = torch.rand(1, 3, 12, 12, 12)
            lt = bcc(coset0, coset1)

    def test_shift_constants(self):
        qc = Lattice("qc")

        coset0 = torch.rand(1, 3, 11, 11)
        coset1 = torch.rand(1, 3, 10, 10)
        lt = qc(coset0, coset1)

        self.assertEqual(tuple(lt.shift_constants(0, 0)), (0, 0))
        self.assertEqual(tuple(lt.shift_constants(0, 1)), (0, 0))
        self.assertEqual(tuple(lt.shift_constants(1, 0)), (-1, -1))
        self.assertEqual(tuple(lt.shift_constants(1, 1)), (0, 0))
