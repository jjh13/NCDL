import torch
import torch.nn as nn
import torch.nn.functional as F


def shift_coset(coset, delta_shift, mode='zero', value=0.):
    preslice = [slice(None), slice(None)] + [slice(abs(min(_, 0)), None) for _ in delta_shift]
    prepad = [[max(_, 0), 0] for _ in delta_shift]
    prepad = sum(reversed(prepad), [])
    coset = coset[preslice]
    if any([_ > 0 for _ in prepad]):
        coset = torch.nn.functional.pad(coset, prepad)
    return coset


def pad_or_slice_like_shape(tensor: torch._tensor, shape, mode='constant', value=0):
    if len(shape) != len(tensor.shape):
        raise ValueError(f"Input tensor must have the same dimension as specified in by the shape")
    dim = len(shape) - 2

    # First, calculate the padding, then pad (if we to)
    right_pad = [b - a for a, b in zip(tensor.shape[:2], shape[:2])]
    if any([_ > 0 for _ in right_pad]):
        right_pad = [[0, 0] if _ < 0 else _ for _ in [right_pad, 0]]
        right_pad = sum(reversed(right_pad), [])
        tensor = F.pad(tensor, pad=right_pad, mode=mode, value=value)

    # Next trim off any unneceesary values
    preslice = [slice(None), slice(None)] + [slice(0, _) for _ in shape[2:]]
    return tensor[preslice]


def common_tensor_slice(tensor_a, tensor_b):
    return [slice(0, min(a, b)) for a, b in zip(tensor_a.shape, tensor_b.shape)]