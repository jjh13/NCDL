![alt text](docs/assets/ncdl_banner.png "tle")

----

### Pre-release Version
This is a pre-release version that includes most of what was in the paper, 
some implementation details have been temporarily lost (due to merge conflicts)
but should be back soon. A few things
- Modules and experiments for further work have been remove (as such, 
  some tests are broken, most pass, still.)
- Mox/atrous pooling is missing its cuda implementation.
- The setup.py's requirement list is messed up, this is a small issue that 
  will be fixed relatively quickly.
  
Changes are currently happening in master, but we'll eventually switch to a more 
sane development schedule.


### Installing / Testing
First ensure the requirements are installed, they're rather minimal
```
pip install -r requirements.txt
```
Make sure PyTorch is installed, too. There's no requirement for this 
in the `requirements.txt` file, but that's because we're relatively 
version agnostic.

Finally, in the root of this directory, install to the local environment
with
```
pip install  -e .
```

### Examples
Check our RTD page, ncdl.ai. there are some gaps, but
the key functionality is documented, and there are relatively comprehensive 
examples in the `examples` directory. Please take a look at those.

Documentation is in `docs/build/index.html`. It's worth looking at for some 
clarity on what's currently available in NCDL. 

[comment]: <> (NTS: Autosummary doesn't seem to be pulling the most recent docstrings
try to fix this before the final revision)

See `modules/autoencoder.py`. It has a number of models defined. Some are 
successful experiments that didn't make it into the current version of the 
paper, some are less successful. That's a full worked neural network implemented
with NCDL.

### Experiments
Experiments are in the `experiments` folder. You will need pytorch lighting 1.8.1 
to run them. The configs are in the `configs` folder (you'll also need the.
