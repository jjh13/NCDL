#NCDL: Non-Cartesian Deep Learning

---
### Introduction
NCDL is a library for performing tensor operations and tracking their gradients 
 on non-Cartesian lattices. Non-Cartesian lattices, such as the hexagonal or BCC/FCC, 
lattices can be particularly useful for representing data with a natural circular structure, such as 
images or scientific simulations. However, traditional tensor operations, which are 
based on the Cartesian coordinate system, can be difficult to apply to non-Cartesian data.

Our library helps solve this problem by providing a comprehensive set of tools 
for performing tensor computations on integer non-Cartesian lattices. These 
tools can be used to build and train machine learning models on non-Cartesian 
data, allowing you to take advantage of the unique properties and structure 
of your data. With the NCDL, you can easily perform common machine learning operations, 
such as convolution, on non-Cartesian data, enabling you to build more powerful and accurate machine learning models.

### Philosophy
The general philosophy for NCDL is to remain as PyTorch-onic as possible. That is, we try to adhere
to the general structure and abstractions that PyTorch already uses. The base abstraction we introduce is the
`LatticeTensor`. See [LatticeTensors.md](LatticeTensors.md) for details; but effectively a `LatticeTensor` generalizes the notion 
of a tensor on a (uniform) non-Cartesian grid. This abstraction complicates things a little:
- Arithmetic on `LatticeTensors` is not exactly trivial. There's some ambiguity is their representation, so element-wise
 arithmetic slightly more involved; again, see [Design.md](LatticeTensors.md).
- Padding, which is another fundamental operation on tensors, is also more involved. Since we represent lattice tensors
as multiple interleaved grids, we have to individually pad each coset grid; but also take care to ensure that the 
  resulting lattice tensor remain valid. See [CosetPadding.md](CosetPadding.md) for details.
- Up/down sampling is also, unsurprisingly, more complicated. We tackle this in a general setting, allowing for a user 
to simply specify a decimation/dilation matrix. See [Resampling.md](Resampling.md) for details.
- Convolution also needs to be adapted to multiple grids. WE do this by splitting the convolutiomn sum into separate 
convolutions over (smaller) Cartesian grids. See [Convolution.md](Convolution.md) for details.

Most of these operations are relatively easy to understand with simple examples, but are somewhat annoying to 
implement.