import unittest
import torch
from ncdl.lattice import Lattice
from ncdl.util.stencil import Stencil


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
            (0, 0), (2, 0), (0, 2), (2, 2), (1, 1)
        ], qc)

        decomposition = stencil.coset_decompose(packed_output=False)

        self.assertEqual(len(decomposition[0]), 4)
        self.assertTrue(all([_ in decomposition[0] for _ in [(0,0),(2,0),(0,2),(2,2)]]))

        self.assertEqual(len(decomposition[1]), 1)
        self.assertTrue(all([_ in decomposition[1] for _ in [(1,1)]]))

        decomposition = stencil.coset_decompose(packed_output=True)

        self.assertEqual(len(decomposition[0]), 4)
        self.assertTrue(all([_ in decomposition[0] for _ in [(0,0),(1,0),(0,1),(1,1)]]))

        self.assertEqual(len(decomposition[1]), 1)
        self.assertTrue(all([_ in decomposition[1] for _ in [(0,0)]]))

    def test_stencil_construction_deltas(self):
        qc = Lattice("qc")
        stencil = Stencil([
            (0, 0), (0, 2), (2, 2), (1, 1)
        ], qc)

        lt = qc({
            (0, 0): torch.rand(1, 4, 16, 16),
            (-1, 1): torch.rand(1, 4, 16, 16)
        })

        self.assertTupleEqual(stencil.delta_shift(lt, 0, 0), (0, 0))
        self.assertTupleEqual(stencil.delta_shift(lt, 0, 1), (-1, 0))
        self.assertTupleEqual(stencil.delta_shift(lt, 1, 0), (0, 0))
        self.assertTupleEqual(stencil.delta_shift(lt, 1, 1), (0, -1))

    def test_stencil_decompose(self):
        qc = Lattice("qc")
        stencil = Stencil([
            (0, 0), (0, 2), (2, 2), (1, 1)
        ], qc)

        # Get the indices for the weight tensor
        index, rindex = stencil.weight_index(0)

        # Create a new index and assign it with some data
        wts = stencil.zero_weights(0, 1, 1)
        wts[:, :, rindex[(0, 0)]] = 0.5
        wts[:, :, rindex[(0, 1)]] = 1.5
        wts[:, :, rindex[(1, 1)]] = 2.5

        # Check that the correct locations unpack to the correct weights
        upacked = stencil.unpack_weights(0, wts, torch.IntTensor(index))
        self.assertEqual(upacked[0, 0, 0, 0], 0.5)
        self.assertEqual(upacked[0, 0, 0, 1], 1.5)
        self.assertEqual(upacked[0, 0, 1, 1], 2.5)
        self.assertEqual(upacked[0, 0, 1, 0], 0.0)

    def test_stencil_construction_padding(self):
        qc = Lattice("qc")
        stencil = Stencil([
            (0, 0), (0, 2), (2, 2), (1, 1)
        ], qc, center=(1, 1))

        lt = qc({
            (0, 0): torch.rand(1, 4, 16, 16),
            (-1, 1): torch.rand(1, 4, 16, 16)
        })

        padding = stencil.padding_for_lattice_tensor(lt)
        for _ in padding:
            self.assertEqual(_, 1)

        lt = stencil.pad_lattice_tensor(lt)
        self.assertEqual(lt.coset_vectors[1][0].item(), 1)
        self.assertEqual(lt.coset_vectors[1][1].item(), -1)

        self.assertEqual(lt.coset(0).shape[-1], 17)
        self.assertEqual(lt.coset(0).shape[-2], 17)
        self.assertEqual(lt.coset(1).shape[-1], 17)
        self.assertEqual(lt.coset(1).shape[-2], 17)


    def test_stencil_decompose_padded(self):
        qc = Lattice("qc")
        stencil = Stencil([
            (2, 2), (2, 4), (4, 4), (3, 3), (3, 1), (1, 3)
        ], qc, center=(2, 2))

        lt = qc({
            (0, 0): torch.rand(1, 4, 16, 16),
            (-1, 1): torch.rand(1, 4, 16, 16)
        })

        self.assertTupleEqual(stencil.delta_shift(lt, 0, 0), (-1,-1))
        self.assertTupleEqual(stencil.delta_shift(lt, 0, 1), (-1, 0))
        self.assertTupleEqual(stencil.delta_shift(lt, 1, 0), (-1,-1))
        self.assertTupleEqual(stencil.delta_shift(lt, 1, 1), (0, -1))


        # Get the indices for the weight tensor
        index, rindex = stencil.weight_index(0)

        # Create a new index and assign it with some data
        wts = stencil.zero_weights(0, 1, 1)
        wts[:, :, rindex[(0, 0)]] = 0.5
        wts[:, :, rindex[(0, 1)]] = 1.5
        wts[:, :, rindex[(1, 1)]] = 2.5

        # Check that the correct locations unpack to the correct weights
        upacked = stencil.unpack_weights(0, wts, torch.IntTensor(index))
        self.assertEqual(upacked[0, 0, 0, 0], 0.5)
        self.assertEqual(upacked[0, 0, 0, 1], 1.5)
        self.assertEqual(upacked[0, 0, 1, 1], 2.5)
        self.assertEqual(upacked[0, 0, 1, 0], 0.0)