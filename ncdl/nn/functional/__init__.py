import sys
import torch.nn.functional as F
from ncdl.lattice import LatticeTensor

__import_from = [
    "threshold", "threshold_", "relu_",
    "relu", "hardtanh", "hardtanh_", "hardswish", "relu6",
    "elu", "elu_", "selu", "celu",
    "leaky_relu", "leaky_relu_", "rrelu", "rrelu_",
    "glu", "gelu", "logsigmoid", "hardshrink",
    "tanhshrink", "softsign", "softplus", "softmin",
    "softmax", "softshrink", "gumbel_softmax", "log_softmax",
    "tanh", "sigmoid", "hardsigmoid", "silu", "mish"]

def _extend_to_lattice_tensor(functional):
    function = getattr(F, functional)
    def wrapped_curried_function(lt: LatticeTensor, *args, **kwargs):
        return lt.parent({
            v: function(*[lt.coset(idx), *args], **kwargs) for idx,v in enumerate(lt.coset_vectors)
        })
    sys.modules[__name__].__setattr__(functional, wrapped_curried_function)
[_extend_to_lattice_tensor(_) for _ in __import_from]

