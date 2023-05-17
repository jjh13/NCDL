import torch
import math
from torch import nn
from torch.nn import init
from ncdl.lattice import LatticeTensor, Lattice
from ncdl.nn.functional.convolution import lattice_conv
# from ncdl.util.stencil import Stencil


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
                 stencil: "Stencil",
                 groups: int = 1,
                 bias: bool = True):
        """

        :param lattice:
        :param channels_in:
        """
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

        # Decompose the stencil and create the parameter tensor
        coset_stencils = stencil.coset_decompose(packed_output=True)
        stencil_offset = 0
        self.stencil_info = []

        for coset_id, stencil in enumerate(coset_stencils):
            if len(stencil) == 0:
                self.stencil_info += [(stencil_offset, 0)]
                continue

            if not self._stencil.is_coset_square(coset_id):
                stencil_index, _ = self._stencil.weight_index(coset_id)
                self.register_buffer(f"stencil_index_{coset_id}", torch.IntTensor(stencil_index))

            self.stencil_info += [(stencil_offset, len(stencil))]
            stencil_offset += len(stencil)

        self.weights = nn.Parameter(torch.empty(channels_out, channels_in, stencil_offset))
        if bias is True:
            self._bias = nn.Parameter(torch.empty(channels_out))
        self.reset_parameters()
        #
        # """
        # Setup the weights for each coset's stencil. Note that the
        # """
        # coset_stencils = stencil.coset_decompose(packed_output=True)
        # for coset_id, stencil in enumerate(coset_stencils):
        #     if len(stencil) == 0:
        #         continue
        #     # Regardless of the stencil type, we zero out the stencils here
        #     weights = nn.Parameter(torch.empty(channels_out, channels_in//channels_in, len(stencil)))
        #
        #     if not self._stencil.is_coset_square(coset_id):
        #         stencil_index, _ = self._stencil.weight_index(coset_id)
        #         self.register_buffer(f"stencil_index_{coset_id}", torch.IntTensor(stencil_index))
        #     self.register_parameter(f"weight_{coset_id}", weights)
        # if bias is True:
        #     self._bias = nn.Parameter(torch.empty(channels_out))
        # self.reset_parameters()

    def reset_parameters(self) -> None:
        k = len(self._stencil.stencil) * self._channels_out
        init.uniform_(self.weights, -1/math.sqrt(k), 1/math.sqrt(k))

        if self._bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weights)
            if fan_in != 0:
                bound = 1 / math.sqrt(fan_in)
                init.uniform_(self._bias, -bound, bound)

    def get_convolution_weights(self, coset_id):
        """
        Returns the /square/ convolution weights for this coset. Internally, conv weights are stored in params of
        shape (c,k,len(stencil), this needs to be unpacked into a 4-d tensor. The Stencil class handles this, padding
        in extra zeros if neccesary.
        """

        slice_offset, slice_size = self.stencil_info[coset_id]
        if slice_size == 0:
            return None

        weights = self.weights[:, :, slice_offset:slice_offset+slice_size]
        index = None

        if not self._stencil.is_coset_square(coset_id):
            index = self.get_buffer(f"stencil_index_{coset_id}")
        return self._stencil.unpack_weights(coset_id, weights, index)

    def forward(self, lt: LatticeTensor) -> LatticeTensor:
        if lt.parent != self._lattice:
            raise ValueError("Input lattice tensor must belong to the correct lattice!")

        weights = [self.get_convolution_weights(i) for i in range(self._lattice.coset_count)]
        return lattice_conv(lt, self._lattice, self._stencil, weights, self._bias, self._groups)


class LatticeStyleConv(nn.Module):
    def __init__(self,
                 lattice: Lattice,
                 channels_in: int,
                 channels_out: int,
                 stencil: "Stencil",
                 demod: bool = True,
                 bias: bool = True,
                 eps: float = 1e-6):
        """

        :param lattice:
        :param channels_in:
        """
        super().__init__()

        # Typecheck all inputs
        type_fails = []
        if not isinstance(lattice, Lattice):
            type_fails += [('lattice', 'Lattice', type(lattice))]

        if not isinstance(channels_in, int):
            type_fails += [('channels_in', 'int', type(channels_in))]

        if not isinstance(channels_out, int):
            type_fails += [('channels_out', 'int', type(channels_out))]

        if not isinstance(bias, bool):
            type_fails += [('bias', 'bool', type(bias))]

        if len(type_fails):
            raise TypeError(
                f"LatticeConvolution instantiated with invalid or malformed parameters: " + ", ".join([
                    f"'{param}' should be of type '{expected}', not '{obs}'" for param, expected, obs in type_fails
                ])
            )
        self._demod = demod
        self._eps = eps
        self._stencil = stencil
        self._lattice = lattice
        self._channels_out = channels_out
        self._channels_in = channels_in
        self._bias = None

        # Decompose the stencil and create the parameter tensor
        coset_stencils = stencil.coset_decompose(packed_output=True)
        stencil_offset = 0
        self.stencil_info = []

        for coset_id, stencil in enumerate(coset_stencils):
            if len(stencil) == 0:
                self.stencil_info += [(stencil_offset, 0)]
                continue

            if not self._stencil.is_coset_square(coset_id):
                stencil_index, _ = self._stencil.weight_index(coset_id)
                self.register_buffer(f"stencil_index_{coset_id}", torch.IntTensor(stencil_index))

            self.stencil_info += [(stencil_offset, len(stencil))]
            stencil_offset += len(stencil)

        self.weights = nn.Parameter(torch.empty(channels_out, channels_in, stencil_offset))
        if bias is True:
            self._bias = nn.Parameter(torch.empty(channels_out))
        self.reset_parameters()

    def get_convolution_weights(self, coset_id, weights_packed):
        """
        Returns the /square/ convolution weights for this coset. Internally, conv weights are stored in params of
        shape (c,k,len(stencil), this needs to be unpacked into a 4-d tensor. The Stencil class handles this, padding
        in extra zeros if neccesary.
        """

        slice_offset, slice_size = self.stencil_info[coset_id]
        if slice_size == 0:
            return None

        weights = weights_packed[:, :, slice_offset:slice_offset+slice_size]
        index = None
        if not self._stencil.is_coset_square(coset_id):
            index = self.get_buffer(f"stencil_index_{coset_id}")
        return self._stencil.unpack_weights(coset_id, weights, index)

    def reset_parameters(self) -> None:
        k = len(self._stencil.stencil) * self._channels_out
        init.uniform_(self.weights, -1/math.sqrt(k), 1/math.sqrt(k))

        if self._bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weights)
            if fan_in != 0:
                bound = 1 / math.sqrt(fan_in)
                init.uniform_(self._bias, -bound, bound)

    def forward(self, lt: LatticeTensor, y: torch.Tensor):
        """
        Forward function for styleconv on a lattice tensor.

        :param lt: Input lattice tensor. Should have inner dimensions [bs, channels_in]
        :param y: Inpuy modulation weights, should be a tensor of dimension [bs, channels_in]

        :return: A new lattice tensor with [bs, channels_out] *.
        """
        batch_size, c, *_ = lt.coset(0).shape
        assert batch_size == y.shape[0]

        w1 = y[:, None, :, None]        # [BS, 1,   in, 1]
        w2 = self.weights[None, :, :, :] # [1,  out, in, h*w*...]
        weights = w2 * (w1 + 1)         # [BS, out, in, h*w*...]

        if self._demod:
            d = torch.rsqrt((weights ** 2).sum(dim=(2, 3), keepdim=True) + self._eps)
            weights = weights * d

        # Reshape weights -- we push the batch index into the channel indices
        # this allows us to use the built-in convolution functions with a different set
        # of weights per batch element
        _, _, *ws = weights.shape
        weights = weights.reshape(batch_size * self._channels_out, *ws) # [bs*weights, in, h*w*...]
        cosets = [lt.coset(_).reshape(1, -1, *lt.coset(_).shape[2:]) for _ in range(lt.parent.coset_count)]
        unpacked_weights = [self.get_convolution_weights(_, weights) for _ in range(lt.parent.coset_count)]

        # Do the grouped convolution, then reshape the output
        lt = lattice_conv(lt, self._lattice, self._stencil, unpacked_weights, None, groups=batch_size, alt_cosets=cosets)
        cosets = [lt.coset(_).reshape(batch_size, self._channels_out, *lt.coset(_).shape[2:]) for _ in range(lt.parent.coset_count)]

        # Construct the final l-tensor and return it
        return LatticeTensor(None, parent=lt.parent, alt_cosets=cosets, alt_offsets=lt.coset_vectors)
