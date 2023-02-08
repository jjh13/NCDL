import torch
import torch.nn as nn
from ncdl.lattice import LatticeTensor, Lattice
from ncdl.nn.functional.convolution import lattice_conv
from ncdl.util.stencil import Stencil


class LatticeConvolution(nn.Module):
    """
    LatticeConvolution

    Implements the ``convolution'' (technically, cross-correlation) operation for LatticeTensors. This interface is
    meant to be as similar as possible to nn.Conv2d. However, we don't support dilation and strided convolution. It's
    possible to do both of these, however the implementation is very intricate (we need to fuse both the downsampling
    and conv operations). Currently, simply use ncdl.nn.functional.downsample to get the downsample operation.

    We also don't support transposed convolution. Mainly because there's no need. The only real benefit to to transposed
    convs is that they increase resolution. You can acheive the same effect (with more flexibility) with
    ncdl.nn.functional.upsample followed by a lattice conv.

    We also don't directly support dilation -- if you want to dilate your filter, then simply dilate your filter before
    instantiating this object.
    """

    def __init__(self,
                 lattice: Lattice,
                 channels_in: int,
                 channels_out: int,
                 stencil: Stencil,
                 groups: int = 1,
                 bias: bool = True):
        super().__init__()

        # Typecheck all inputs
        type_fails = []
        if not isinstance(lattice, Lattice):
            type_fails += [('lattice', 'Lattice', type(lattice))]

        if not isinstance(channels_in, int):
            type_fails += [('channels_in', 'int', type(channels_in))]

        if not isinstance(channels_out, int):
            type_fails += [('channels_out', 'int', type(channels_out))]

        if not isinstance(groups, int):
            type_fails += [('groups', 'int', type(groups))]

        if not isinstance(bias, bool):
            type_fails += [('bias', 'bool', type(bias))]

        if len(type_fails):
            print(type_fails)
            raise TypeError(
                f"LatticeConvolution instantiated with invalid or malformed parameters: " + ", ".join([
                    f"'{param}' should be of type '{expected}', not '{obs}'" for param, expected, obs in type_fails
                ])
            )

        if channels_in % groups != 0:
            raise ValueError("Input channel number must be divisible by 'groups'")

        self._stencil = stencil
        self._lattice = lattice
        self._channels_out = channels_out
        self._channels_in = channels_in
        self._groups = groups
        self._bias = None

        """
        Setup the weights for each coset's stencil. Note that the 
        """
        coset_stencils = stencil.coset_decompose(packed_output=True)
        for coset_id, stencil in enumerate(coset_stencils):
            # Regardless of the stencil type, we zero out the stencils here
            weights = nn.Parameter(torch.empty(channels_out, channels_in, len(stencil)))

            if not self._stencil.is_coset_square(coset_id):
                stencil_index = self._stencil.weight_index(coset_id)
                self.register_buffer(f"stencil_index_{coset_id}", torch.IntTensor(stencil_index))
            self.register_parameter(f"weight_{coset_id}", weights)
        if bias is True:
            self._bias = nn.Parameter(torch.empty(channels_out))

    def get_convolution_weights(self, coset_id):
        """
        Returns the /square/ convolution weights for this coset. Internally, conv weights are stored in params of
        shape (c,k,len(stencil), this needs to be unpacked into a 4-d tensor. The Stencil class handles this, padding
        in extra zeros if neccesary.
        """
        weights = self.get_parameter(f"weight_{coset_id}")
        index = None
        if not self._stencil.is_coset_square(coset_id):
            index = self.get_buffer(f"stencil_index_{coset_id}")
        return self._stencil.unpack_weights(coset_id, weights, index)

    def forward(self, lt: LatticeTensor) -> LatticeTensor:
        weights = [self.get_convolution_weights(i) for i in range(self._lattice.coset_count)]
        return lattice_conv(lt, self._lattice, self._stencil, weights, self._bias, self._groups)
