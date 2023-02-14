import unittest
import torch
import torch.nn as nn
import ncdl.nn as ncnn
from ncdl.lattice import Lattice


class LatticeConstruction(unittest.TestCase):
    def setUp(self):
        self.devices = [torch.device('cpu')]
        if torch.cuda.is_available():
            self.devices += [torch.device('cuda:0')]
        # elif torch.backends.mps.is_available():
        #     self.devices += [torch.device('mps:0')]

    def activation_test(self, partial_nc, partial_c, device):
        qc = Lattice("qc")
        a0_ = torch.rand(1, 3, 16, 16, device=device)
        a1_ = torch.rand(1, 3, 16, 16, device=device)

        a0 = a0_.detach().clone()
        a1 = a1_.detach().clone()
        a0.requires_grad = True
        a1.requires_grad = True

        lta = qc({
            (0, 0): a0 - 0.5,
            (-1, -1): a1 - 0.5
        })

        leaky_relu = partial_nc()
        lto = leaky_relu(lta)

        result = lto.coset(0) + lto.coset(1)

        result = (2.0 - result.sum()) ** 2
        result.backward()

        # Cartesian
        a0c = a0_.detach().clone()
        a1c = a1_.detach().clone()
        a0c.requires_grad = True
        a1c.requires_grad = True
        leaky_reluc = partial_c()
        result_c = leaky_reluc(a0c - 0.5) + leaky_reluc(a1c - 0.5)
        result_c = (2.0 - result_c.sum()) ** 2
        result_c.backward()

        with torch.no_grad():
            self.assertAlmostEqual(((a0.grad - a0c.grad) ** 2).sum().item(), 0)
            self.assertAlmostEqual(((a1.grad - a1c.grad) ** 2).sum().item(), 0)


    def test_act_layers(self):
        """
        Tests that lattices, when constructed with consistent devices, report the same device.
        """
        for device in self.devices:
            self.activation_test(ncnn.ReLU, nn.ReLU, device)
            self.activation_test(ncnn.LeakyReLU, nn.LeakyReLU, device)
            self.activation_test(ncnn.SELU, nn.SELU, device)
            self.activation_test(ncnn.SiLU, nn.SiLU, device)
            self.activation_test(ncnn.ReLU6, nn.ReLU6, device)

    def test_act_broken(self):
        """
        Tests that lattices, when constructed with consistent devices, report the same device.
        """
        for device in self.devices:
            self.activation_test(ncnn.PReLU, nn.PReLU, device)
            self.activation_test(ncnn.RReLU, nn.RReLU, device)
