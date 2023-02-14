import unittest
import torch
import numpy as np
from ncdl.lattice import Lattice
from ncdl.nn.functional.pad import pad


class LatticePaddingTests(unittest.TestCase):
    def setUp(self):
        self.devices = [torch.device('cpu')]
        if torch.cuda.is_available():
            self.devices += [torch.device('cuda:0')]
        elif torch.backends.mps.is_available():
            self.devices += [torch.device('mps:0')]

    def test_lattice_pad_3l(self):
        offset, scale = [
            np.array([0, 0], dtype='int'),
            np.array([1, 1], dtype='int'),
            np.array([2, 2], dtype='int'),
        ], np.array([3, 3], dtype='int')

        qc = Lattice(offset, scale)
        for device in self.devices:
            lt = qc(
                torch.rand(1, 3, 3, 3),
                torch.rand(1, 3, 3, 3),
                torch.rand(1, 3, 3, 3)
            ).to(device)

            lt = pad(lt, (1, 0, 1, 0))
            bounds = lt.lattice_bounds()
            self.assertEqual(bounds[2], (-1, 8))
            self.assertEqual(bounds[3], (-1, 8))
            self.assertEqual(lt.device, device)

            lt = pad(lt, (0, 1, 0, 1))
            bounds = lt.lattice_bounds()
            self.assertEqual(bounds[2], (-1, 9))
            self.assertEqual(bounds[3], (-1, 9))
            self.assertEqual(lt.device, device)

            lt = pad(lt, (1, 0, 1, 0))
            bounds = lt.lattice_bounds()
            self.assertEqual(bounds[2], (-2, 9))
            self.assertEqual(bounds[3], (-2, 9))
            self.assertEqual(lt.device, device)

            lt = pad(lt, (3, 0, 3, 0))
            bounds = lt.lattice_bounds()
            self.assertEqual(bounds[2], (-2, 12))
            self.assertEqual(bounds[3], (-2, 12))
            self.assertEqual(lt.device, device)

            lt = pad(lt, (1, 0, 1, 0))
            bounds = lt.lattice_bounds()
            self.assertEqual(bounds[2], (0, 15))
            self.assertEqual(bounds[3], (0, 15))
            self.assertEqual(lt.device, device)


    def test_lattice_pad_qc(self):
        qc = Lattice("qc")
        for device in self.devices:

            lt = qc(
                torch.rand(1, 3, 3, 3),
                torch.rand(1, 3, 3, 3)
            ).to(device)

            lt = pad(lt, (1,0,1,0))
            bounds = lt.lattice_bounds()
            self.assertEqual(bounds[2],  (-1, 5))
            self.assertEqual(bounds[3],  (-1, 5))
            self.assertEqual(lt.device,  device)

            lt = pad(lt, (0,1,0,1))
            bounds = lt.lattice_bounds()
            self.assertEqual(bounds[2],  (-1, 6))
            self.assertEqual(bounds[3],  (-1, 6))
            self.assertEqual(lt.device,  device)

            lt = pad(lt, (2,0,2,0))
            bounds = lt.lattice_bounds()
            self.assertEqual(bounds[2],  (-1, 8))
            self.assertEqual(bounds[3],  (-1, 8))
            self.assertEqual(lt.device,  device)

            lt = pad(lt, (1,0,1,0))
            bounds = lt.lattice_bounds()
            self.assertEqual(bounds[2],  (0, 10))
            self.assertEqual(bounds[3],  (0, 10))
            self.assertEqual(lt.device,  device)


    def test_lattice_pad_bcc(self):
        qc = Lattice("BCC")
        for device in self.devices:

            lt = qc(
                torch.rand(1, 3, 3, 3, 3),
                torch.rand(1, 3, 3, 3, 3),
            ).to(device)

            lt = pad(lt, (1,0,1,0,1,0))
            bounds = lt.lattice_bounds()
            self.assertEqual(bounds[2],  (-1, 5))
            self.assertEqual(bounds[3],  (-1, 5))
            self.assertEqual(bounds[4],  (-1, 5))
            self.assertEqual(lt.device,  device)

            lt = pad(lt, (0,1,0,1,0,1))
            bounds = lt.lattice_bounds()
            self.assertEqual(bounds[2],  (-1, 6))
            self.assertEqual(bounds[3],  (-1, 6))
            self.assertEqual(bounds[4],  (-1, 6))
            self.assertEqual(lt.device,  device)

            lt = pad(lt, (2,0,2,0,2,0))
            bounds = lt.lattice_bounds()
            self.assertEqual(bounds[2],  (-1, 8))
            self.assertEqual(bounds[3],  (-1, 8))
            self.assertEqual(bounds[4],  (-1, 8))
            self.assertEqual(lt.device,  device)

            lt = pad(lt, (1,0,1,0,1,0))
            bounds = lt.lattice_bounds()
            self.assertEqual(bounds[2],  (0, 10))
            self.assertEqual(bounds[3],  (0, 10))
            self.assertEqual(bounds[4],  (0, 10))
            self.assertEqual(lt.device,  device)
