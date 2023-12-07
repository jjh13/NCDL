![alt text](docs/assets/ncdl_banner.png "tle")

----
### What is NCDL?
NCDL is a software package for differentiable computation on 
non-Cartesian lattices. But what is a "non-Cartesian lattice"?
Have you ever wondered why pixels are square? Why not use Hexagons
instead; the neighbor structure of a hexagon is more uniform than
a square. Notably, using a hexagonal structure has a higher capacity
for information compared to a square structure [1,2,3]. There are 
similar structures in higher dimensions that allow 

This package provides a means of performing computation on these 
structures. We implement these operations in PyTorch, and can therefore
provide differentiable computation for no additional development cost.

It is worth mentioning that this is a research codebase. As such, our focus 
is heavily weighted towards correctness.


### Initial Release Notes
This is the inital release of NCDL. The core functionality of the lattice tensor,
plus all of the operations mentioned in the paper are implemented. Most network architecture
structures that are defined on Cartesian data can be relatively easily moved
to a non-Cartesian sapce.

There have been a few things cut out of this version that have still not been merged back into
the repo:
- Max/atrous pooling is missing its cuda implementation.
  
Changes are currently happening in master, but we'll eventually switch to a more 
sane development practice.


### Installing / Testing
First, make sure PyTorch is installed. There's no specific requirement for this 
in the `setup.py` file; this is because NCDL is relatively agnostic to the version 
of PyTorch used. 

Next, clone the repository via
```bash
git clone https://github.com/jjh13/NCDL.git
```

Finally, in the root of this directory, install to the local environment
with
```
pip install  -e .
```

### Examples and Documentation
Check our readthedocs page, https://ncdl.ai. Currently, the key functionality is documented,
but there may be gaps. Please open an issue if you find any documentation lacking.
There are relatively comprehensive examples in the `examples` directory. Please take a look at those.

Documentation is in `docs/build/index.html`. It's worth looking at for some 
clarity on what's currently available in NCDL. 

See `modules/autoencoder.py`. It has a number of models defined. Some are 
successful experiments that didn't make it into the current version of the 
paper, some are less successful. That's a full worked neural network implemented
with NCDL.

### Experiments
Experiments are in the `experiments` folder. You will need pytorch lighting 1.8.1 
to run them. The configs are in the `configs` folder (you'll also need the.

### Citing this Work
If you use this work in your publication, be sure to cite
```
@inproceedings{horacsek2023ncdl,
  title={NCDL: A Framework for Deep Learning on non-Cartesian Lattices},
  author={Horacsek, Joshua John and Alim, Usman},
  booktitle={Thirty-seventh Conference on Neural Information Processing Systems},
  year={2023}
}
```