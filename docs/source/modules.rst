
Layer API
================================

We define some basic layers to mimic the workflows already present
in PyTorch. These layers


Utility Layers
****************

.. autosummary::
   :toctree: _autosummary

   ncdl.nn.LatticeWrap
   ncdl.nn.LatticeUnwrap
   ncdl.nn.LatticePad

Convolution Layer
****************

.. autosummary::
   :toctree: _autosummary

   ncdl.nn.LatticeConvolution

Pooling Layers
****************

.. autosummary::
   :toctree: _autosummary

   ncdl.nn.LatticeMaxPooling

Resampling Layers
******************

.. autosummary::
   :toctree: _autosummary

   ncdl.nn.LatticeDownsample
   ncdl.nn.LatticeUpsample

Activation Layers
******************

.. autosummary::
   :toctree: _autosummary

   ncdl.nn.ReLU
   ncdl.nn.ReLU6
   ncdl.nn.RReLU
   ncdl.nn.PReLU
   ncdl.nn.LeakyReLU
   ncdl.nn.Hardtanh
   ncdl.nn.Tanh
   ncdl.nn.Tanhshrink
   ncdl.nn.Threshold
   ncdl.nn.Softmax
   ncdl.nn.Softmin
   ncdl.nn.Softsign
   ncdl.nn.Softplus
   ncdl.nn.SELU
   ncdl.nn.Sigmoid
   ncdl.nn.SiLU
   ncdl.nn.Softshrink
   ncdl.nn.Mish
   ncdl.nn.Hardswish
   ncdl.nn.Hardshrink
   ncdl.nn.Hardsigmoid


Normalization Layers
**********************

.. autosummary::
   :toctree: _autosummary

   ncdl.nn.LatticeBatchNorm
   ncdl.nn.LatticeGroupNorm
   ncdl.nn.LatticeInstanceNorm