import unittest
import torch
from ncdl.lattice import Lattice
from ncdl.util.stencil import Stencil
from ncdl.nn import LatticeConvolution, LatticeDownsample, LatticePad
from ncdl.nn.modules.convolution import LatticeStyleConv

import numpy as np
import matplotlib.pyplot as plt

import ncdl.nn as ncnn
import torch.nn as nn

class ConvolutionTests(unittest.TestCase):
    def setUp(self):
        if torch.cuda.is_available():
            self.device = torch.device('cuda:0')
        elif torch.backends.mps.is_available():
            self.device = torch.device('mps:0')
        else:
            self.device = torch.device('cpu')

    def test_shape_out_lsc(self):
        qc = Lattice('qc')
        stencil = Stencil([
            (0,0),(0,2),(2,2),(1,1),(2,0)
        ], qc)

        lt = qc(
            { (0,0): torch.rand(2, 4, 3, 3),
              (-1,1): torch.rand(2, 4, 4, 3)
              }
        )

        lc = LatticeStyleConv(qc, 4, 8, stencil, bias=True)
        # print(lc._calc_shape_out(lt, 0))
        # print(lc._calc_shape_out(lt, 1))

        lc.forward(lt, torch.rand(2, 4))

    def test_shape_out(self):
        qc = Lattice('qc')
        stencil = Stencil([
            (0,0),(0,2),(2,2),(1,1),(2,0)
        ], qc)

        lt = qc(
            { (0,0): torch.rand(1, 4, 3, 3),
              (-1,1): torch.rand(1, 4, 4, 3)
              }
        )

        lc = LatticeConvolution(qc, 1, 1, stencil, groups=1, bias=True)
        print(lc._calc_shape_out(lt, 0))
        print(lc._calc_shape_out(lt, 1))

        lc.forward(lt)


    def test_conv(self):
        qc = Lattice('qc')
        stencil = Stencil([
            (1,1),(1,3),(3,3),(2,2),(3,1)
        ], qc, center=(1,1))
        #
        # lt = qc(
        #     { (0,0): torch.rand(1, 4, 2, 2),
        #       (-1,1): torch.rand(1, 4, 3, 2)
        #       }
        # )

        z = torch.zeros(1, 1, 5, 4)
        a = torch.zeros(1, 1, 4, 4)
        a[0,0,2,2] = 1.0
        z[0,0,2,1] = 1.0
        lt = qc(
            { (0,0): a,
              (-1,1): z
              }
        )

        lc = LatticeConvolution(qc, 1, 1, stencil, groups=1, bias=False)

        #lt = stencil.pad_lattice_tensor(lt)

        index, rindex = stencil.weight_index(1)
        wts = stencil.zero_weights(1, 1, 1)
        wts[:,:,rindex[(0,0)]] = 0.1
        wts[:,:,rindex[(0,1)]] = 0.2
        wts[:,:,rindex[(1,0)]] = 0.3
        wts[:,:,rindex[(1,1)]] = 0.4
        from torch import nn
        lc.__setattr__("weight_1", nn.Parameter(wts))


        index, rindex = stencil.weight_index(0)
        wts = stencil.zero_weights(0, 1, 1)
        wts[:,:,rindex[(0,0)]] = 0.5
        lc.__setattr__("weight_0", nn.Parameter(wts))


        # print(lc._calc_shape_out(lt, 0))
        # print(lc._calc_shape_out(lt, 1))
        xx = lc.forward(lt)

        # Fixing random state for reproducibility

        xlist, ylist,slist = [], [], []
        for x in range(-1, 10):
            for y in range(-1, 10):
                if not xx.on_lattice(torch.IntTensor([x,y])):
                    continue

                xlist += [x]
                ylist += [y]

                if x % 2 == 0 or y % 2 == 0:
                    s = xx.coset(0)[:, :, x//2, y//2]
                else:
                    s = xx.coset(1)[:, :, (x+1)//2, (y-1)//2]
                slist += [(s.item() + 0.05)*100]
                # print(slist)

        plt.scatter(xlist, ylist, s=slist, alpha=0.5)
        plt.show()

        print(xx.coset(0))
        print(xx.coset(1))

        pass

    def test_conv_grad(self):
        qc = Lattice('qc')
        stencil = Stencil([
            (0,0),(0,2),(2,2),(1,1),(2,0)
        ], qc, center=(1,1))

        a = torch.rand(1, 4, 10, 10, requires_grad=True)
        b = torch.rand(1, 4, 10, 10, requires_grad=True)

        lt = qc(
            { (0,0): a,
              (1,1): b
              }
        )

        # lc = torch.nn.Sequential(
            # LatticePad(qc, stencil),
        lc0 = LatticeConvolution(qc, 4, 1, stencil, groups=1, bias=True)
        # )
        lc = torch.nn.Sequential(
            LatticePad(qc, stencil),
            lc0,
            LatticeDownsample(qc, np.array([[1, 1], [1, -1]]))
        )

        out = lc(lt)
        out = out.coset(0).sum() # + out.coset(1).sum()
        out.backward()

        pass





    def test_conv(self):
        qc = Lattice('qc')
        stencil = Stencil([
            (1,1),(1,3),(3,3),(2,2),(3,1)
        ], qc, center=(1,1))
        #
        # lt = qc(
        #     { (0,0): torch.rand(1, 4, 2, 2),
        #       (-1,1): torch.rand(1, 4, 3, 2)
        #       }
        # )

        z = torch.zeros(1, 1, 5, 4)
        a = torch.zeros(1, 1, 4, 4)
        a[0,0,2,2] = 1.0
        z[0,0,2,1] = 1.0
        lt = qc(
            { (0,0): a,
              (-1,1): z
              }
        )


        lc = LatticeConvolution(qc, 1, 1, stencil, groups=1, bias=False)
        #lt = stencil.pad_lattice_tensor(lt)

        index, rindex = stencil.weight_index(1)
        wts = stencil.zero_weights(1, 1, 1)
        wts[:,:,rindex[(0,0)]] = 0.1
        wts[:,:,rindex[(0,1)]] = 0.2
        wts[:,:,rindex[(1,0)]] = 0.3
        wts[:,:,rindex[(1,1)]] = 0.4
        from torch import nn
        lc.__setattr__("weight_1", nn.Parameter(wts))


        index, rindex = stencil.weight_index(0)
        wts = stencil.zero_weights(0, 1, 1)
        wts[:,:,rindex[(0,0)]] = 0.5
        lc.__setattr__("weight_0", nn.Parameter(wts))


        # print(lc._calc_shape_out(lt, 0))
        # print(lc._calc_shape_out(lt, 1))
        xx = lc.forward(lt)

        # Fixing random state for reproducibility

        xlist, ylist,slist = [], [], []
        for x in range(-1, 10):
            for y in range(-1, 10):
                if not xx.on_lattice(torch.IntTensor([x,y])):
                    continue

                xlist += [x]
                ylist += [y]

                if x % 2 == 0 or y % 2 == 0:
                    s = xx.coset(0)[:, :, x//2, y//2]
                else:
                    s = xx.coset(1)[:, :, (x+1)//2, (y-1)//2]
                slist += [(s.item() + 0.05)*100]
                # print(slist)

        plt.scatter(xlist, ylist, s=slist, alpha=0.5)
        plt.show()

        print(xx.coset(0))
        print(xx.coset(1))

        pass



    def test_conv_smooth(self):
        qc = Lattice('qc')
        cp = Lattice('cp')

        qc_d = LatticeDownsample(qc, torch.IntTensor([[1, 1], [1, -1]]))
        cp_d = LatticeDownsample(cp, torch.IntTensor([[1, 1], [1, -1]]))

        stencil = Stencil([
            (0,0),(2,0),(1,1),(0,2),(2,2)
        ], qc, center=(1,1))


        stencil_cp = Stencil(
            [(0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2), (2, 0), (2, 1), (2, 2)],
            cp, center=(1,1)
        )

        z = torch.zeros(1, 1, 9, 9)
        a = torch.zeros(1, 1, 10, 10)
        a[0,0,2,2] = 1.0
        z[0,0,2,1] = 0
        lt = qc(
            { (0,0): a,
              (1,1): z
              }
        )

        low_pass_qc = LatticeConvolution(qc, 1, 1, stencil, groups=1, bias=False)
        low_pass_cp = LatticeConvolution(cp, 1, 1, stencil_cp, groups=1, bias=False)

        #lt = stencil.pad_lattice_tensor(lt)

        index, rindex = stencil.weight_index(0)
        wts = stencil.zero_weights(0, 1, 1)
        wts[:,:,rindex[(0,0)]] = 0.125
        wts[:,:,rindex[(0,1)]] = 0.125
        wts[:,:,rindex[(1,0)]] = 0.125
        wts[:,:,rindex[(1,1)]] = 0.125
        from torch import nn
        low_pass_qc.__setattr__("weight_0", nn.Parameter(wts))


        index, rindex = stencil.weight_index(1)
        wts = stencil.zero_weights(1, 1, 1)
        wts[:,:,rindex[(0,0)]] = 0.5
        low_pass_qc.__setattr__("weight_1", nn.Parameter(wts))


        index, rindex = stencil_cp.weight_index(0)
        wts = stencil_cp.zero_weights(0, 1, 1)
        wts[:,:,rindex[(0,1)]] = 0.125
        wts[:,:,rindex[(1,0)]] = 0.125
        wts[:,:,rindex[(2,1)]] = 0.125
        wts[:,:,rindex[(1,2)]] = 0.125
        wts[:,:,rindex[(1,1)]] = 0.5
        from torch import nn
        low_pass_cp.__setattr__("weight_0", nn.Parameter(wts))
        # -------------------------

        # print(lc._calc_shape_out(lt, 0))
        # print(lc._calc_shape_out(lt, 1))
        ds = nn.Sequential(
            LatticePad(qc, stencil),
            low_pass_qc,
            LatticePad(qc, stencil),
            low_pass_qc,
            LatticePad(qc, stencil),
            low_pass_qc,
            LatticePad(qc, stencil),
            low_pass_qc,
            LatticePad(qc, stencil),
            low_pass_qc,
            LatticePad(qc, stencil),
            low_pass_qc,
            # qc_d,
            # LatticePad(cp, stencil_cp),
            # low_pass_cp,
            # cp_d
        )
        xx = ds(lt)


        # Fixing random state for reproducibility

        xlist, ylist,slist = [], [], []
        for x in range(-1, 10):
            for y in range(-1, 10):
                if not xx.on_lattice(np.array([x,y], dtype='int')):
                    continue

                xlist += [x]
                ylist += [y]

                if x % 2 == 0 or y % 2 == 0:
                    s = xx.coset(0)[:, :, x//2, y//2]
                else:
                    s = xx.coset(1)[:, :, (x-1)//2, (y-1)//2]
                slist += [(s.item() + 0.0)*100]
                # print(slist)

        plt.scatter(xlist, ylist, s=slist, alpha=0.5)
        plt.show()

        print(xx.coset(0))
        print(xx.coset(1))

        pass

    def test_backward(self):
        lattice = Lattice('cp')
        stencil = Stencil([
            (0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2), (2, 0), (2, 1), (2, 2)
        ], lattice, center=(1, 1))

        lc = LatticeConvolution(lattice, 32, 16, stencil, groups=1, bias=False)

        conv_proxy = nn.Sequential(
            ncnn.LatticeWrap(),
            lc,
            ncnn.LatticeUnwrap()
        )
        input_tensor = torch.rand(1, 32, 64, 64, requires_grad=True)

        output = conv_proxy(input_tensor)

        loss = (output.sum() - 1)**2
        loss.backward()

        tensor_copy = input_tensor.detach().clone()
        tensor_copy.requires_grad = True
        tconv = nn.Conv2d(32, 16, kernel_size=3, padding=0, bias=False)
        tconv.weight = nn.Parameter(
            lc.get_convolution_weights(0).detach().clone()
        )

        output = tconv(tensor_copy)
        loss = (output.sum() - 1)**2
        loss.backward()
        pass


    def test_hexcov(self):
        lattice = Lattice('qc')
        stencil = Stencil([
            (1, 1), (2, 0), (2, 2), (3, 1), (4, 0), (4, 2), (5, 1)
        ], lattice, center=(1, 1))

        output = stencil.coset_decompose(True)

        lc = LatticeConvolution(lattice, 32, 16, stencil, groups=1, bias=False)


        pass






