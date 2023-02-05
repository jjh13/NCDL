import unittest
import torch
from ncdl.lattice import Lattice
from ncdl.util.stencil import Stencil

from ncdl.nn.functional.downsample import downsample
from ncdl.nn.functional.upsample import upsample
from ncdl.nn.functional.pad import pad

class StencilTests(unittest.TestCase):
    def setUp(self):
        if torch.cuda.is_available():
            self.device = torch.device('cuda:0')
        elif torch.backends.mps.is_available():
            self.device = torch.device('mps:0')
        else:
            self.device = torch.device('cpu')

    def test_stencil_construction(self):
        qc = Lattice("qc")
        stencil = Stencil([
            (0,0),(2,0),(0,2),(2,2),(1,1)
        ], qc)

        decomposition = stencil.coset_decompose(output_cartesian=False)

        self.assertEqual(len(decomposition[0]), 4)
        self.assertTrue(all([_ in decomposition[0] for _ in [(0,0),(2,0),(0,2),(2,2)]]))

        self.assertEqual(len(decomposition[1]), 1)
        self.assertTrue(all([_ in decomposition[1] for _ in [(1,1)]]))

        decomposition = stencil.coset_decompose(output_cartesian=True)

        self.assertEqual(len(decomposition[0]), 4)
        self.assertTrue(all([_ in decomposition[0] for _ in [(0,0),(1,0),(0,1),(1,1)]]))

        self.assertEqual(len(decomposition[1]), 1)
        self.assertTrue(all([_ in decomposition[1] for _ in [(0,0)]]))



    def test_stencil_construction(self):
        qc = Lattice("qc")
        stencil = Stencil([
            (0,0),(0,2),(2,2),(1,1)
        ], qc)

        lt = qc(
            { (0,0): torch.rand(1, 4, 16, 16),
              (-1,1): torch.rand(1, 4, 16, 16)
              }
        )

        print(stencil.delta_shift(lt, 0, 0))
        print(stencil.delta_shift(lt, 0, 1))
        print(stencil.delta_shift(lt, 1, 0))
        print(stencil.delta_shift(lt, 1, 1))

        print(stencil.coset_decompose(True))

        index, rindex = stencil.weight_index(0)
        print(rindex)

        wts = stencil.zero_weights(0, 1, 1)
        wts[:,:,rindex[(0,0)]] = 0.5
        wts[:,:,rindex[(0,1)]] = 1.5
        wts[:,:,rindex[(1,1)]] = 2.5

        print(wts.shape, wts)
        print(stencil.unpack_weights(0, wts, torch.IntTensor(index) ))
        z = torch.zeros(1,1,2,2)
        z[0,0,0,0] = 0.5
        z[0,0,0,1] = 1.5
        z[0,0,1,1] = 2.5
        print(z)