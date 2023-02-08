import unittest
import torch
from ncdl.lattice import Lattice
from ncdl.nn.functional.downsample import downsample, downsample_lattice
from ncdl.nn.functional.upsample import upsample
from ncdl.nn.modules.resample import LatticeUpsample, LatticeDownsample


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
            [torch.IntTensor([0,0])], torch.IntTensor([1,1])
        )
        a0 = torch.rand(1, 3, 10, 10)
        lt = cl({(0,0):a0})
        print(lt)


    def test_downsample_layer(self):
        qc = Lattice("qc")

        D = torch.IntTensor([[1, 1], [1, -1]])

        downsample_lattice(qc, D)

        a0 = torch.rand(1, 3, 16, 16)
        a1 = torch.rand(1, 3, 16, 16)

        lta = qc({
            (0, 0): a0,
            (1, 1): a1
        })
        D = torch.IntTensor([[1,1],[1,-1]])
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


    def test_upsample_layer(self):
        qc = Lattice("qc")

        D = torch.IntTensor([[1, 1], [1, -1]])

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






