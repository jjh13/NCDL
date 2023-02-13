from typing import Tuple, List
import torch
import numpy as np

from itertools import product
import math


def is_lattice_site(L, s) -> bool:
    L = np.array(L)
    s = np.array(s)
    sp = L @ np.linalg.solve(L, s).round()
    err = np.abs(sp - s).sum()
    tf = err < 1e-5
    return tf

def find_coset_vectors(L):
    """
    Finds the

    Let me be explicitly clear here. This is a terrible algorithm for this problem -- it's only somewhat acceptable
    because the dimension of the problem is so low. If we were operating in higher dimensions, I would prefer the
    diamond cutting algorithm -- it implicitly gives all the information we need here. I'll live with this

    """
    assert L.shape[0] == L.shape[1]
    s = L.shape[0]

    b = np.abs(np.linalg.det(L))
    assert b != 0.0 and b != 0
    D = np.array([[0] * s] * s)
    bounds = []
    for dim in range(s):
        for c in range(1, int(b + 1)):
            v = np.array([0 if _ != dim else c for _ in range(s)])
            if is_lattice_site(L, v):
                D[dim, dim] = c
                bounds += [(0, c - 1)]
                break

    v_list = []
    for pt in product(*bounds):
        if all([_ == 0 for _ in pt]):
            continue

        if is_lattice_site(L, pt):
            v_list += [pt]
    D = np.array(D)
    return D, [tuple([0] * s)] + list(set(v_list))

def _gcd_star(*args):
    gcd = 0
    for a in args:
        gcd = math.gcd(gcd, a)
    return gcd

def interval_differences(list_of_intervals):
    """
    Given a list of intervals, this returns a set of intervals that are covered by one or less elements from the input
    intervals
    """
    min_i = min([a for a, _ in list_of_intervals])
    max_i = max([b for _, b in list_of_intervals])
    list_of_intervals = list_of_intervals + [(min_i, max_i)]
    pos_cnt = {ipos: 0 for ipos in sum([list(_) for _ in list_of_intervals], [])}
    for (a, b) in list_of_intervals:
        pos_cnt[a] += 1
        pos_cnt[b] -= 1
    divs = sorted(pos_cnt.keys())
    csum = 0
    ret = []
    interval_start = None
    for k in divs:
        csum += pos_cnt[k]
        if csum <= 2 and interval_start is None:
            interval_start = k
            continue
        if csum >= 3 and interval_start is not None:
            ret += [(interval_start, k)]
            interval_start = None
    if interval_start is not None:
        ret += [(interval_start, k)]
    return ret


def get_coset_vector_from_name(name: str) -> Tuple[List[torch.IntTensor], torch.Tensor]:
    """
    For a canonical

    :param name:
    :return: a tuple of torch tensor
    """
    name = name.lower()

    if name == "bcc" or name == "body centered cubic":
        return [
            torch.IntTensor([0, 0, 0]),
            torch.IntTensor([1, 1, 1]),
        ], torch.tensor([2., 2., 2.])

    elif name == "fcc" or name == "face centered cubic":
        return [
            torch.IntTensor([0, 0, 0]),
            torch.IntTensor([0, 1, 1]),
            torch.IntTensor([1, 0, 1]),
            torch.IntTensor([1, 1, 0]),
        ], torch.tensor([2., 2., 2.])

    elif name == "qc" or name == "quincunx":
        return [
            torch.IntTensor([0, 0]),
            torch.IntTensor([1, 1]),
        ], torch.tensor([2., 2.])

    elif name == "cp" or name == "cartesian_planar":
        return [
            torch.IntTensor([0, 0]),
        ], torch.tensor([1., 1.])

    raise ValueError(f"Lattice '{name}' is not valid?")


def hnf_transform(lattice):
    pass

def get_matrix_from_mod_basis(lattice_vectors):
    # arrange lattice vectors then get HNF
    pass

def get_mod_basis(lattice):
    pass
    # # put lattice in HNF
    # lattice = hnf_transform(lattice)
    #
    # # do a stupid trick to extract the
    #
    # return D, [v1,vk]

# @torch.no_grad
# def get_subsampled_mod_basis(lattice_tensor, S):
#     L = get_matrix_from_mod_basis(lattice_tensor)
#     D = L @ S
#     D = hnf_transform(L)
#
#     D, v_list = get_mod_basis(D)
