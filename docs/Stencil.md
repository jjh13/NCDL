# Stencil 
--

The `Stencil` class is an abstraction for filters with specific geometry. It's
used to create and manipulate weights for filters. 

### Stencil construction
To construct a stencil, do the following
```python
from ncdl.util.stencil import Stencil

stencil = Stencil([
    (2, 2), (2, 4), (4, 4), (3, 3), (3, 1), (1, 3)
], lattice_instance, center=(2, 2))
```
this constructs a stencil. You can then use this stencil in a LatticeConv layer
to define the geometry of the filter. The stencil will validate that the stencil points belong to the lattice.


### Padding with a stencil
If you specified center in the stencil construction, you can also pad with 
```python
stencil.pad_lattice_tensor(lattice_tensor)
```
which will effectively do "same" padding. That is, after you convolve with a filter given with "stencil" the output
lattice tensor will have the same sizes as the input lattice tensor.

### Convolution
Convolution is relatively simple. You probably to create a convolution layer as

```python
from ncdl.nn.modules import LatticeConvolution

lconv1 = LatticeConvolution(lattice, channels_in, channels_out, stencil)
```
you would then use
```python
lattice_tensor = stencil.pad_lattice_tensor(lattice_tensor)
lattice_tensor = lconv1(lattice_tensor)
```
to perform the convolution.


### Pooling
TODO: Pooling is not currently supported, but will be soon-ish