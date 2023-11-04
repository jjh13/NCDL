import torch
import unittest
from ncdl import Lattice, Stencil
from ncdl.nn.functional.pooling import lattice_maxpool
from ncdl.extensions.atrous_pooling import MaxPoolAtrousFunction


class LatticeMaxPool(unittest.TestCase):
    def setUp(self):
        self.devices = [torch.device('cpu')]
        if torch.cuda.is_available():
            self.devices += [torch.device('cuda:0')]
        elif torch.backends.mps.is_available():
            self.devices += [torch.device('mps:0')]

    def test_atrous_pooling(self):
        """
        Tests that the forward and backward functions of the atrous pooling operation
        are consistent with pytorch's base implementation.
        """
        cartesian_stencil = [
            [0,0],
            [0,1],
            [1,0],
            [1,1]
        ]

        x = torch.randn(1, 3, 6, 6, requires_grad=True)
        rout = torch.randn(1, 3, 5, 5, requires_grad=True)

        x1 = x.clone().detach()
        x2 = x.clone().detach()
        x1.requires_grad = True
        x2.requires_grad = True

        rout1 = rout.detach().clone()
        rout2 = rout.detach().clone()

        output1 = MaxPoolAtrousFunction.apply(x1, cartesian_stencil)
        output2 = torch.nn.functional.max_pool2d(x2, kernel_size=(2,2), stride=(1,1))

        with torch.no_grad():
            l2 = torch.nn.functional.mse_loss(output1, output2)
            self.assertAlmostEqual(l2.item(), 0.0)

        z1 = (output1 - rout1) ** 2
        z2 = (output2 - rout2) ** 2
        loss1 = z1.sum()
        loss2 = z2.sum()

        loss1.backward()
        loss2.backward()

        with torch.no_grad():
            l2 = torch.nn.functional.mse_loss(x1.grad, x2.grad)
            self.assertAlmostEqual(l2.item(), 0.0)

    def test_max_pool(self):
        """
        Tests that operations between shifted (but consistent) lattices succeed
        """
        qc = Lattice("qc")

        stencil = Stencil([
            (0,0), (0,2), (2,2), (1,1), (2,0)
        ], qc)

        a0 = torch.zeros(1, 1, 5, 5)
        a1 = torch.zeros(1, 1, 5, 5)

        a0[:,:,1,1] = 2.0
        a0[:,:,1,2] = 3.0
        a0[:,:,2,1] = 4.0
        a0[:,:,2,2] = 5.0

        lta = qc({
            ( 0,  0): a0,
            (-1, -1): a1
        })
        coset0_expected = \
            torch.tensor([[[[2., 3., 3., 0.],
                          [4., 5., 5., 0.],
                          [4., 5., 5., 0.],
                          [0., 0., 0., 0.]]]])

        coset1_expected = \
            torch.tensor([[[[0., 0., 0., 0.],
                          [0., 2., 3., 0.],
                          [0., 4., 5., 0.],
                          [0., 0., 0., 0.]]]])

        lto = lattice_maxpool(lta, qc, stencil)

        self.assertAlmostEqual(torch.nn.functional.mse_loss(lto.coset(0), coset0_expected).item(), 0.0)
        self.assertAlmostEqual(torch.nn.functional.mse_loss(lto.coset(1), coset1_expected).item(), 0.0)









