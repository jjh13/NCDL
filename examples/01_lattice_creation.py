import torch
from ncdl.lattice import *


# Creating a lattice tensor is simple, first create a lattice factory
# this object represents
qc = Lattice("qc")

# We easily create a lattice tensor -- you don't need to specify the lattice
# shifts; by default the lattice shifts will be in the positive quadrant
lt = qc({
    (0, 0): torch.rand(1, 3, 128, 128),
    (1, 1): torch.rand(1, 3, 128, 128)
})

# You can perform arithmetic as usual
lt_res = lt + lt

# You can get the individual cosets via the coset method
print(lt_res.coset(0))
