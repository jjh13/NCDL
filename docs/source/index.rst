.. NCDL documentation master file.

NCDL Docs
================================
NCDL is a light-weight package over top of PyTorch that adds the
ability to work on (regular) non-Cartesian grids. If you're not
familiar with the term 'non-Cartesian grid', a good introductory
example is the Hexagonal grid. In 3-dimensions, further
examples are the FCC, BCC, FCO etc... (and even more in higher
dimensions).

Philosphy
*************
The general philosophy for NCDL is to remain as generic and as
PyTorch-onic as possible. That is, we try to adhere to the general structure and
abstractions that PyTorch already uses. We perfer generality
over performance. As time progresses, we hope to improve performance
and get to speed parity with equivatent Cartesian approaches.


Getting Started
****************
Refer to the `README.md` in the root of the github repository. You
will need /some/ version of PyTorch (I have not done extensive
testing regarding which version work, but I know in the range
`[1.8.1, 2.0.0]` is working for me.

Key Concepts
*************
**Lattice**:
A Lattice is a discrete point structure that forms a group. Think of
it like this, imagine an image with square pixels. The "lattice" is
the underlying point structure of the center of those pixels. Lattices
in NCDL are factory like objects, they describe the lattice geometry,
but do not hold data. They construct LatticeTensors, which hold
data (and keep track of boundary information).

*Further Information*:

.. toctree::
   :maxdepth: 1

   lattice

**LatticeTensor**:
Lattice tensors are the de-facto standard datastructure in NCDL. All
operations in NCDL operate on lattice tensor.

*Further Information*:

.. toctree::
   :maxdepth: 1

   lattice_tensor


**Stencil**:
Stencils simply record the geometry of a filter to be used with
convolution and/or max pooling.

*Further Information*:

.. toctree::
   :maxdepth: 1

   stencil

Examples
************
To make the ideas in this documentation more concrete, we provide a small set
of examples. These come in the form of Jupyter notebooks, you can find these
in the `examples` directory of the base github repository.

 * Constructing Lattices and Lattice Tensors
 * Lattice Tensor Arithmetic
 * Upsampling and Downsampling
 * Constructing Stencils
 * Using Stencils
 * Using the Layer API

Layer API
************
In the same vein as PyTorch, we provide a "layer" abstraction for modules.
These are intended to be used in the same way as base PyTorch layers, with
the exception that these take lattice tensors instead of base tensors.

.. toctree::
   :maxdepth: 1

   modules


Functional API
***************
The functional API contains the base implementations for the Layer API.
For many cases, you very likely want to use the Layer API.

.. toctree::
   :maxdepth: 1

   functional


Examples
************
Examples in the form of Jupyter notebooks are in the examples folder in the
base of the github repository. These cover the key concepts in this
documentation. Please ensure that you have installed NCDL in your
local environment before running these.


Full API Reference
**********************
The full API reference is here

.. toctree::
   :maxdepth: 1

   api

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
