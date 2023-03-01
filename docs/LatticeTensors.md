# Design

---
The key ideology of NCDL is to keep code as close as possible to vanilla PyTorch. Two fundamental 
abstractions we present are the `Lattice`  and  `LatticeTensor` classes. `Lattice` objects simply
keep track of integer lattice structures. Think of them as factories --- they create `LatticeTensor` objects.
Generally you don't want to create `LatticeTensors` on your own.

Almost every other abstraction or method (LatticeTensors, Stencils, Layers, etc...) require a `Lattice` object in their 
constructor. Sometimes this is functional (a layer may need to be aware of the explicit lattice structure it is supposed 
to operate over) and sometimes it is organizational (many objects simply check that a given lattice tensor belongs on 
its expected Lattice structure).

Lattices are represented by interleaved Cartesian grids. This is general enough for any integer lattice 
(and a non-Integer lattice can be handled with a diagonal scale, however this is an abstraction the user must provide). 
You create a Lattice in a few different ways, for examples, we can make a Quincunx lattice via
```python
from ncdl.lattice import Lattice
qc = Lattice("quincunx")
```
Typically, it's easiest to specify it as a shorthand string. You can also explicitly
give the coset structure.
```python
from ncdl.lattice import Lattice
import numpy as np
qc = Lattice([
            np.array([0, 0], dtype='int'),
            np.array([1, 1], dtype='int'),
        ], np.array([2, 2], dtype='int'))
```
Which is, perhaps, a little less elegant. You can also register new lattices if you find it more convenient. 


### LatticeTensors
We've defined the lattice structures over which we want to operate, but they currrently don't hold any data
As we implied earlier, we don't intend for you to explicitly create `LatticeTensors`; most of the time you will
want to use the `Lattice` instance as a factory. For example:
```python
import torch
lt = qc(
    torch.rand(1, 3, 4, 4), 
    torch.rand(1, 3, 3, 3)
)
```
creates a lattice tensor. Lattice tensors have a slight oddity about them. It's easiest to illustrate this with 
a picture:

In general it is possible for grids to interleave in many ways. This is an oddity specific to LatticeTensors that 
we must take care to implement. If we don't allow this, padding becomes less robust (we would be limited to certain 
types of padding). 
```python
import torch
lt = qc(
    {
        (0,0): torch.rand(1, 3, 3, 3), 
        (-1,-1): torch.rand(1, 3, 4, 4)
    }
)
```