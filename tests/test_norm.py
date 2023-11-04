import unittest
import torch
from ncdl.lattice import Lattice
from ncdl.nn.functional.norm import coset_moments, instance_norm, batch_norm, group_norm
from ncdl.nn import LatticeBatchNorm, LatticeInstanceNorm, LatticeGroupNorm

# from dataset.duts import SODLoader

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
        _, mean, var, numels = coset_moments(lt, dims=[-1,-2])

        x = torch.cat([coset0, coset1], dim=-1)
        mu = x.mean([-1, -2], keepdim=True)
        va = torch.var(x, dim=[-1, -2], keepdim=True, unbiased=False)
        self.assertTrue((mu - mean).pow(2).sum().item() < 1e-5)
        self.assertTrue((var/numels - va).pow(2).sum().item() < 1e-5)

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

    def test_instance_norm_layer_cp(self):
        cp = Lattice('cp')

        lbn = LatticeInstanceNorm(cp, 16, track_running_stats=True)
        cbn = torch.nn.InstanceNorm2d(16, track_running_stats=True)

        # Run a number of batches to ensure that some stats get accumulated
        for _ in range(2):
            coset0 = torch.rand(2, 16, 16, 16, requires_grad=True)

            # Clone and setup (lattice) tensors
            coset_joined = coset0.clone().detach()
            coset_joined.requires_grad = True
            lt = cp(coset0)

            # Pass thru the layers
            lto = lbn(lt)
            cto = cbn(coset_joined)

            with torch.no_grad():
                # Check that the running stats are computed correctly
                self.assertAlmostEqual((lbn.running_var - cbn.running_var).pow(2).sum().item(), 0)
                self.assertAlmostEqual((lbn.running_mean - cbn.running_mean).pow(2).sum().item(), 0)

                # Check that the output is equal
                self.assertAlmostEqual((lto.coset(0) - cto[:, :, :, ]).pow(2).sum().item(), 0)

            # backward
            (cto**3).sum().backward()
            (lto.coset(0)**3).sum().backward()

            with torch.no_grad():
                self.assertAlmostEqual((coset_joined.grad - coset0.grad).pow(2).sum().item(), 0, places=6)

        # Enter eval mode
        lbn.eval()
        cbn.eval()

        # Go thru more batches to ensure that the running stats are identical
        for _ in range(2):
            coset0 = torch.rand(2, 16, 16, 16, requires_grad=True)

            # Clone and setup (lattice) tensors
            coset_joined = coset0.clone().detach()
            coset_joined.requires_grad = True
            lt = cp(coset0)

            # Pass thru the layers
            lto = lbn(lt)
            cto = cbn(coset_joined)

            with torch.no_grad():
                self.assertAlmostEqual((lbn.running_var - cbn.running_var).pow(2).sum().item(), 0)
                self.assertAlmostEqual((lbn.running_mean - cbn.running_mean).pow(2).sum().item(), 0)


                # First check that the batch norm agrees on the cosets
                self.assertAlmostEqual((lto.coset(0) - cto[:, :, :, :16]).pow(2).sum().item(), 0)

            # backward
            (cto**3).sum().backward()
            (lto.coset(0)**3).sum().backward()

            with torch.no_grad():
                self.assertAlmostEqual((coset_joined.grad - coset0.grad).pow(2).sum().item(), 0, places=6)


    def test_batch_norm(self):
        qc = Lattice('qc')
        coset0 = torch.rand(2, 3, 16, 16)*100
        coset1 = torch.rand(2, 3, 16, 16)*0.001
        lt = qc(coset0, coset1)
        x = torch.cat([coset0, coset1], dim=-1)

        ltn = batch_norm(lt)
        xn = torch.nn.functional.batch_norm(x, None, None, training=True)

        self.assertTrue((ltn.coset(0) - xn[:,:,:,:16]).pow(2).sum() < 1e-5)
        self.assertTrue((ltn.coset(1) - xn[:,:,:,16:]).pow(2).sum() < 1e-5)



    def test_batch_norm_layer(self):
        cp = Lattice('cp')

        lbn = LatticeBatchNorm(cp, 16)
        cbn = torch.nn.BatchNorm2d(16)

        for _ in range(4):
            coset0 = torch.rand(16, 16, 16, 16, requires_grad=True)
            # coset1 = torch.rand(2, 16, 16, 16, requires_grad=True)
            coset_joined = coset0.clone().detach()
            coset_joined.requires_grad = True

            lt = cp(coset0)

            lto = lbn(lt)
            cto = cbn(coset_joined)

            # # First check that the batch norm agrees on the cosets
            self.assertAlmostEqual((lto.coset(0) - cto[:, :, :, :16]).pow(2).sum().item(), 0)
            # self.assertAlmostEqual((lto.coset(0) - cto[:, :, :, :16]).pow(2).sum().item(), 0)
            # self.assertTrue((lto.coset(1) - cto[:, :, :, 16:]).pow(2).sum().item() < 1e-5)


    def test_batch_norm_layer_qc(self):
        qc = Lattice('qc')

        lbn = LatticeBatchNorm(qc, 16)
        cbn = torch.nn.BatchNorm2d(16)

        for _ in range(1):
            coset0 = torch.rand(2, 16, 16, 16, requires_grad=True)
            coset1 = torch.rand(2, 16, 16, 16, requires_grad=True)
            coset_joined = torch.cat([coset0, coset1], dim=-1).detach()
            coset_joined.requires_grad = True

            lt = qc(coset0, coset1)

            lto = lbn(lt)
            cto = cbn(coset_joined)

            # First check that the batch norm agrees on the cosets
            self.assertAlmostEqual((lto.coset(0) - cto[:, :, :, :16]).pow(2).sum().item(), 0)
            self.assertAlmostEqual((lto.coset(0) - cto[:, :, :, :16]).pow(2).sum().item(), 0)
            self.assertTrue((lto.coset(1) - cto[:, :, :, 16:]).pow(2).sum().item() < 1e-5)



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
