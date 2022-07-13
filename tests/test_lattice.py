import unittest
import torch
from ncdl.lattice import Lattice


class LatticeConstruction(unittest.TestCase):
    def setUp(self):
        if torch.cuda.is_available():
            self.device = torch.device('cuda:0')
        elif torch.backends.mps.is_available():
            self.device = torch.device('mps:0')
        else:
            self.device = torch.device('cpu')


    def test_create_qc_shifted_add(self):
        qc = Lattice("qc")
        self.assertEqual(qc.dimension, 2)

        a0 = torch.ones(1, 3, 10, 10)
        a1 = torch.zeros(1, 3, 11, 11)

        lta = qc({
            ( 0,  0): a0,
            (-1, -1): a1
        })

        b0 = torch.ones(1, 3, 11, 11)
        b1 = torch.zeros(1, 3, 10, 10)

        ltb = qc({
            (0,  0): b0,
            (1, 1): b1
        })

        ltc = lta + ltb

        print(ltc.coset(0))

    def test_create_qc_add(self):
        qc = Lattice("qc")
        self.assertEqual(qc.dimension, 2)

        a0 = torch.rand(1, 3, 10, 10)
        a1 = torch.rand(1, 3, 9, 9)

        b0 = torch.rand(1, 3, 10, 10)
        b1 = torch.rand(1, 3, 9, 9)

        lta = qc(a0, a1)
        ltb = qc(b0, b1)

        print(lta._find_correspondence(ltb))

        ltc = lta + ltb

        self.assertEqual(ltc.coset(0).sum(), (a0+b0).sum())
        self.assertEqual(ltc.coset(1).sum(), (a1+b1).sum())


    def test_create_qc_mult(self):
        qc = Lattice("qc")
        self.assertEqual(qc.dimension, 2)

        a0 = torch.rand(1, 3, 10, 10)
        a1 = torch.rand(1, 3, 9, 9)

        b0 = torch.rand(1, 3, 10, 10)
        b1 = torch.rand(1, 3, 9, 9)

        lta = qc(a0, a1)
        ltb = qc(b0, b1)

        print(lta._find_correspondence(ltb))

        ltc = lta * ltb

        self.assertEqual(ltc.coset(0).sum(), (a0*b0).sum())
        self.assertEqual(ltc.coset(1).sum(), (a1*b1).sum())



    def test_create_qc(self):
        qc = Lattice("qc")
        self.assertEqual(qc.dimension, 2)

        lt = qc(
            torch.rand(1, 3, 10, 10),
            torch.rand(1, 3, 9, 9)
        )

        lts = lt[:, :, 2:-1, 2:-1]

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
