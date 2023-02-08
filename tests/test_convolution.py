import unittest
import torch
from ncdl.lattice import Lattice
from ncdl.util.stencil import Stencil
from ncdl.nn.modules.convolution import LatticeConvolution

import matplotlib.pyplot as plt


class ConvolutionTests(unittest.TestCase):
    def setUp(self):
        if torch.cuda.is_available():
            self.device = torch.device('cuda:0')
        elif torch.backends.mps.is_available():
            self.device = torch.device('mps:0')
        else:
            self.device = torch.device('cpu')

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




