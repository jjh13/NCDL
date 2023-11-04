import os

import torch
from typing import List
from torch.autograd import Function
from torch.utils.cpp_extension import load

max_pool_atrous = None

class MaxPoolAtrousFunction(Function):
    @staticmethod
    def forward(ctx, input, stencil):
        global max_pool_atrous
        if max_pool_atrous is None:
            path = os.path.dirname(os.path.abspath(__file__))
            max_pool_atrous = load(
                name="max_pool_atrous",
                sources=[os.path.join(path, "atrous_pooling/max_pool.cpp")]
            )

        ctx.stencil = stencil
        ctx.save_for_backward(input)
        return max_pool_atrous.forward(input, stencil)

    @staticmethod
    def backward(ctx, grad_y):
        global max_pool_atrous
        return (max_pool_atrous.backward(grad_y, ctx.saved_tensors[0], ctx.stencil), None)


#
# path = os.path.dirname(os.path.abspath(__file__))
# max_pool_atrous = load(name="max_pool_atrous", sources=[os.path.join(path, "atrous_pooling/max_pool.cpp")])