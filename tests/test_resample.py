import unittest
import torch
from ncdl.lattice import Lattice
from ncdl.nn.functional.downsample import downsample, downsample_lattice
from ncdl.nn.functional.upsample import upsample
from ncdl.nn import LatticeUpsample, LatticeDownsample
from ncdl.nn import LatticeConvolution, LatticePad
from ncdl.util.stencil import Stencil
import torch.nn as nn

import numpy as np


def create_smoothing_qc(chan_in):
    qc = Lattice('qc')

    stencil = Stencil([
        (0,0),(2,0),(1,1),(0,2),(2,2)
    ], qc, center=(1,1))
    low_pass_qc = LatticeConvolution(qc, chan_in, chan_in, stencil, groups=3, bias=False)

    index, rindex = stencil.weight_index(0)
    wts = stencil.zero_weights(0, chan_in, 1)
    wts[:, :, rindex[(0, 0)]] = 0.125
    wts[:, :, rindex[(0, 1)]] = 0.125
    wts[:, :, rindex[(1, 0)]] = 0.125
    wts[:, :, rindex[(1, 1)]] = 0.125
    low_pass_qc.__setattr__("weight_0", nn.Parameter(wts))

    index, rindex = stencil.weight_index(1)
    wts = stencil.zero_weights(1, chan_in, 1)
    wts[:, :, rindex[(0, 0)]] = 0.5
    low_pass_qc.__setattr__("weight_1", nn.Parameter(wts))

    return nn.Sequential(
        LatticePad(qc, stencil),
        low_pass_qc
    )

def create_smoothing_qc2(chan_in):
    qc = Lattice('qc')

    stencil = Stencil([
        (1,1),(3,1),(2,2),(1,3),(3,3),(0,2),(2,0),(4,2),(2,4)
    ], qc, center=(2,2))
    low_pass_qc = LatticeConvolution(qc, chan_in, chan_in, stencil, groups=1, bias=False)

    return nn.Sequential(
        LatticePad(qc, stencil),
        low_pass_qc
    )


def create_smoothing_cp(chan_in):
    cp = Lattice('cp')

    stencil = Stencil(
        [(0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2), (2, 0), (2, 1), (2, 2)],
        cp, center=(1, 1)
    )

    low_pass_cp = LatticeConvolution(cp, chan_in, chan_in, stencil, groups=3, bias=False)

    index, rindex = stencil.weight_index(0)
    wts = stencil.zero_weights(0, chan_in, 1)
    wts[:, :, rindex[(0, 1)]] = 0.125
    wts[:, :, rindex[(1, 0)]] = 0.125
    wts[:, :, rindex[(2, 1)]] = 0.125
    wts[:, :, rindex[(1, 2)]] = 0.125
    wts[:, :, rindex[(1, 1)]] = 0.5
    low_pass_cp.__setattr__("weight_0", nn.Parameter(wts))

    return nn.Sequential(
        LatticePad(cp, stencil),
        low_pass_cp
    )

class LatticeConstruction(unittest.TestCase):
    def setUp(self):
        if torch.cuda.is_available():
            self.device = torch.device('cuda:0')
        elif torch.backends.mps.is_available():
            self.device = torch.device('mps:0')
        else:
            self.device = torch.device('cpu')

    def test_create_basic_cartesian(self):
        cl = Lattice(
            [np.array([0,0], dtype='int')], np.array([1,1], dtype='int')
        )
        a0 = torch.rand(1, 3, 10, 10)
        lt = cl({(0,0):a0})
        print(lt)

    def test_downsample_qc2(self):
        qc = Lattice("qc")

        D = np.array([[2, 0], [0, 2]], dtype='int')

        blue_coset = torch.rand(4, 3, 16, 16)
        orange_coset = torch.rand(4, 3, 16, 16)

        lt = qc({
            (0, 0): blue_coset,
            (1, 1): orange_coset
        })

        x = downsample(lt, D)




    def test_downsample_layer(self):
        qc = Lattice("qc")

        D = np.array([[1, 1], [1, -1]], dtype='int')

        downsample_lattice(qc, D)

        a0 = torch.rand(1, 3, 16, 16)
        a1 = torch.rand(1, 3, 16, 16)

        lta = qc({
            (0, 0): a0,
            (1, 1): a1
        })

        ldl = LatticeDownsample(qc, D)
        self.assertNotEqual(qc, ldl.down_lattice)

        ldl2 = LatticeDownsample(ldl.down_lattice, D)
        self.assertEqual(qc, ldl2.down_lattice)

        lt = ldl(lta)
        lt = ldl2(lt)
        self.assertTupleEqual(
            tuple(lt.coset(0).shape),
            (1,3,8,8)
        )
        self.assertTupleEqual(
            tuple(lt.coset(1).shape),
            (1,3,8,8)
        )


    def test_smallest_conv(self):
        qc = Lattice("qc")

        D = np.array([[1, 1], [1, -1]], dtype='int')

        downsample_lattice(qc, D)

        a0 = torch.rand(1, 3, 1, 1)
        a1 = torch.rand(1, 3, 1, 1)

        lta = qc({
            (0, 0): a0,
            (1, 1): a1
        })

        sf = create_smoothing_qc2(3)

        ldl = LatticeDownsample(qc, D)
        self.assertNotEqual(qc, ldl.down_lattice)

        lto = sf(lta)
        lt_down = ldl(lto)
        pass


    def test_upsample_layer(self):
        qc = Lattice("qc")

        D = np.array([[1, 1], [1, -1]], dtype='int')

        downsample_lattice(qc, D)

        a0 = torch.rand(1, 3, 16, 16)
        a1 = torch.rand(1, 3, 16, 16)

        lta = qc({
            (0, 0): a0,
            (1, 1): a1
        })
        D = torch.IntTensor([[1,1],[1,-1]])
        ldl = LatticeUpsample(qc, D)
        self.assertNotEqual(qc, ldl.up_lattice)

        ldl2 = LatticeUpsample(ldl.up_lattice, D)
        self.assertEqual(qc, ldl2.up_lattice)

        lt = ldl(lta)
        lt = ldl2(lt)
        self.assertTupleEqual(
            tuple(lt.coset(0).shape),
            (1,3,32,32)
        )
        self.assertTupleEqual(
            tuple(lt.coset(1).shape),
            (1,3,31,31)
        )


    def test_downsample(self):
        qc = Lattice("qc")

        D = torch.IntTensor([[1,1],[1,-1]])

        downsample_lattice(qc, D)

        a0 = torch.rand(1, 3, 10, 10)
        a1 = torch.rand(1, 3, 11, 11)

        lta = qc({
            ( 0,  0): a0,
            (-1, -1): a1
        })

        x = downsample(lta, D)
        x = downsample(x, D)

        print(x)

    def test_downsample_aliasing(self):
        from PIL import Image
        import torchvision
        from IPython.core.display_functions import display
        from matplotlib.pyplot import imshow

        bricks = Image.open("/Users/joshuahoracsek/aliasing2.png")
        brick_t = torchvision.transforms.functional.to_tensor(bricks)[None, ...]

        qc = Lattice("cp")
        l = qc(brick_t)
        print(brick_t.shape)
        D = torch.IntTensor([[1,1],[1,-1]])

        #
        s = create_smoothing_qc(3)
        sc = create_smoothing_cp(3)
        l = downsample(sc(l), D)
        l = downsample(s(l),  D)

        # #
        l = downsample(sc(l), D)
        l = downsample(s(l), D)

        torchvision.transforms.functional.to_pil_image(brick_t[0, :, ::4, ::4]).save("aliasing_cp.png")

        #
        pil = torchvision.transforms.functional.to_pil_image(l.coset(0)[0])

        pil.save('aliasing.png')
        imshow(np.asarray(pil))
        pass

    def test_upsample(self):
        qc = Lattice("qc")

        a0 = torch.rand(1, 3, 3, 3)
        a1 = torch.rand(1, 3, 3, 3)

        lta = qc({
            ( 0,  0): a0,
            (-1, -1): a1
        })

        x = upsample(lta, torch.IntTensor([[1,1],[1,-1]]))
        x = upsample(x, torch.IntTensor([[1,1],[1,-1]]))

        print(x)






