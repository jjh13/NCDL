import unittest
import torch
from ncdl.lattice import Lattice
from ncdl.nn.functional.norm import coset_moments, instance_norm, batch_norm, group_norm
from ncdl.nn.modules import LatticeBatchNorm
from ncdl.nn.modules import LatticeInstanceNorm
from ncdl.nn.modules import LatticeGroupNorm

class LatticeNorm(unittest.TestCase):
    def setUp(self):
        if torch.cuda.is_available():
            self.device = torch.device('cuda:0')
        elif torch.backends.mps.is_available():
            self.device = torch.device('mps:0')
        else:
            self.device = torch.device('cpu')

    def test_moments_0(self):
        qc = Lattice('qc')
        coset0 = torch.rand(2, 3, 16, 16)
        coset1 = torch.rand(2, 3, 16, 16)

        lt = qc(coset0, coset1)
        mean, var = coset_moments(lt, dims=[-1,-2])

        x = torch.cat([coset0, coset1], dim=-1)
        mu = x.mean([-1, -2], keepdim=True)
        va = torch.var(x, dim=[-1, -2], keepdim=True, unbiased=False)
        self.assertTrue((mu - mean).pow(2).sum().item() < 1e-5)
        self.assertTrue((var - va).pow(2).sum().item() < 1e-5)

    def test_instance_norm(self):
        qc = Lattice('qc')
        coset0 = torch.rand(2, 3, 16, 16)
        coset1 = torch.rand(2, 3, 16, 16)

        lt = qc(coset0, coset1)

        x = torch.cat([coset0, coset1], dim=-1)

        ltn = instance_norm(lt)
        xn = torch.nn.functional.instance_norm(x)

        self.assertTrue((ltn.coset(0) - xn[:,:,:,:16]).pow(2).sum() < 1e-5)
        self.assertTrue((ltn.coset(1) - xn[:,:,:,16:]).pow(2).sum() < 1e-5)

    def test_instance_norm_layer(self):
        qc = Lattice('qc')
        coset0 = torch.rand(2, 3, 16, 16)
        coset1 = torch.rand(2, 3, 16, 16)

        lt = qc(coset0, coset1)
        lin = LatticeInstanceNorm(qc, 3)

        lt = lin(lt)


    def test_batch_norm(self):
        qc = Lattice('qc')
        coset0 = torch.rand(2, 3, 16, 16)*10000
        coset1 = torch.rand(2, 3, 16, 16)*0.0001

        lt = qc(coset0, coset1)

        x = torch.cat([coset0, coset1], dim=-1)

        ltn = batch_norm(lt)
        xn = torch.nn.functional.batch_norm(x, None, None, training=True)

        self.assertTrue((ltn.coset(0) - xn[:,:,:,:16]).pow(2).sum() < 1e-5)
        self.assertTrue((ltn.coset(1) - xn[:,:,:,16:]).pow(2).sum() < 1e-5)

    def test_batch_norm_layer(self):
        qc = Lattice('qc')
        coset0 = torch.rand(2, 3, 16, 16)*10000
        coset1 = torch.rand(2, 3, 16, 16)*0.0001

        lt = qc(coset0, coset1)

        lbn = LatticeBatchNorm(qc, 3)
        lt  = lbn(lt)

        # self.assertTrue((ltn.coset(1) - xn[:,:,:,16:]).pow(2).sum() < 1e-5)


    def test_group_norm0(self):
        x = torch.rand(2, 6, 2, 2)

        x_n0 = torch.nn.functional.group_norm(x, 2)

        x_g = x.reshape(2,2,3,2,2)
        mu = x_g.mean([-1,-2,-3], keepdim=True)
        var = x_g.var([-1,-2,-3], keepdim=True, unbiased=False) + 1e-5


        x_gn = (x_g - mu)/(var).sqrt()

        x_n1 = x_gn.reshape(2,6,2,2)
        print(x_n0 - x_n1)

    def test_group_norm_layer(self):
        qc = Lattice('qc')
        coset0 = torch.rand(2, 12, 2, 2)
        coset1 = torch.rand(2, 12, 2, 2)

        lt = qc(coset0, coset1)
        lgn = LatticeGroupNorm(qc, 3, 12)

        lt=lgn(lt)



    def test_group_norm(self):
        qc = Lattice('qc')
        coset0 = torch.rand(2, 12, 2, 2)
        coset1 = torch.rand(2, 12, 2, 2)

        lt = qc(coset0, coset1)

        x = torch.cat([coset0, coset1], dim=-1)

        ltn = group_norm(lt, groups=3)
        xn = torch.nn.functional.group_norm(x, 3)

        self.assertTrue((ltn.coset(0) - xn[:,:,:,:2]).pow(2).sum() < 1e-5)
        self.assertTrue((ltn.coset(1) - xn[:,:,:,2:]).pow(2).sum() < 1e-5)
