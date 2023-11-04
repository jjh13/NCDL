import unittest
import torch
import numpy as np
from ncdl.lattice import Lattice


class LatticeConstruction(unittest.TestCase):
    def setUp(self):
        self.devices = [torch.device('cpu')]
        if torch.cuda.is_available():
            self.devices += [torch.device('cuda:0')]
        elif torch.backends.mps.is_available():
            self.devices += [torch.device('mps:0')]

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
            src_offsets = [lta.coset_vectors[idx] for idx in range(lta.parent.coset_count)]
            src_sizes = [lta.coset(idx).shape for idx in range(lta.parent.coset_count)]

            # Move the tensor to the device
            lta = lta.to(device)
            for idx in range(lta.parent.coset_count):
                for src, trg in zip(lta.coset_vectors[idx], src_offsets[idx]):
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

            self.assertTrue(lta.on_lattice(np.array([0, 0], dtype='int')))

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

            self.assertTrue(lta.on_lattice(np.array([-1,-1], dtype='int')))
            self.assertTrue(lta.on_lattice(np.array([1,1], dtype='int')))
            self.assertTrue(lta.on_lattice(np.array([0,2], dtype='int')))
            self.assertFalse(lta.on_lattice(np.array([-1,2], dtype='int')))

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
    #
    # def test_lattice_slice0(self):
    #     qc = Lattice("qc")
    #     self.assertEqual(qc.dimension, 2)
    #
    #     lt = qc(
    #         torch.rand(1, 3, 10, 10),
    #         torch.rand(1, 3, 9, 9)
    #     )
    #
    #     lts = lt[:, :, 2:-1, 2:-1]
    #     print(lts)

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

    def test_lattice_cat(self):
        import ncdl
        qc = Lattice("qc")

        coset0 = torch.rand(1, 3, 10, 10)
        coset1 = torch.rand(1, 3, 10, 10)
        lt_a = qc({
            (0, 0): coset0,
            (1, 1): coset1
        })

        lt_b = qc({
            (0,0): coset0,
            (-1,-1): coset1
        })

        for dim in [0, 1]:
            ltc = ncdl.cat([lt_a, lt_b], dim=dim)
            self.assertAlmostEqual(((ltc.coset(0) - torch.cat([coset0, coset1], dim=dim))**2).sum().item(), 0.0)
            self.assertAlmostEqual(((ltc.coset(1) - torch.cat([coset1, coset0], dim=dim))**2).sum().item(), 0.0)


