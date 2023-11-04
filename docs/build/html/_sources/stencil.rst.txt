
Stencils
================================
The `Stencil` class is an abstraction for filters with specific geometry. It's
used to create and manipulate weights for filters.

Stencil construction
**********************

To construct a stencil, do the following:

.. code-block:: python

    from ncdl import Stencil

    stencil = Stencil([
        (2, 2), (2, 4), (4, 4), (3, 3), (3, 1), (1, 3)
    ], lattice_instance, center=(2, 2))

You can then use this stencil in a LatticeConv or a LatticeMaxPooling layer to define the geometry of the filter. Keep
in mind that the filter geometry must be strictly positive.
The stencil will validate that the stencil points belong to the lattice.


Padding with a stencil
************************
If you specified center in the stencil construction, you can also pad with
the stencil objects.

.. code-block:: python

    from ncdl import Stencil
    stencil.pad_lattice_tensor(lattice_tensor)

which will effectively do "same" padding. That is, after you convolve with a
filter given with "stencil" the output lattice tensor will have the same sizes
as the input lattice tensor.



.. autosummary::
   :toctree: _autosummary

   ncdl.Stencil