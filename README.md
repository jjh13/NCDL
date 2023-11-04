![alt text](docs/assets/ncdl_banner.png "tle")

----

### Review Version
This is a review version with a few things stripped out.
- Examples and documentation have been minimized, this is somewhat to 
  increase anonymity.
- Modules and experiments for further work have been remove (as such, 
  some tests are broken and have been removed)
- Mox pooling has been removed (it requires a separate build process
  because we use a specialized kernel, which may complicate the process 
  of running the code -- experiments in the paper don't need this anyway)

### Installing / Testing
First ensure the requirements are installed, they're rahter minimal
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
Th

See modules/autoencoder.py. It has a number of models defined. Some are 
successful experiments that didn't make it into the current version of the 
paper.

### Experiments
Experiments are in the `experiments` folder. You will need pytorch lighting 1.8.1 
to run them. The configs are in the `configs` folder
