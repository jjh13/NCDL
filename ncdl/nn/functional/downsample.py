import torch
from hsnf import column_style_hermite_normal_form
from ncdl.lattice import Lattice, LatticeTensor
import numpy as np


from itertools import product
import math


def is_lattice_site(L, s):
    L = torch.Tensor(L)
    s = torch.Tensor(s)
    sp = L @ torch.linalg.solve(L, s).round()
    err = torch.abs(sp - s).sum()
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
    D = torch.IntTensor(D)
    return D, [tuple([0] * s)] + list(set(v_list))

def _gcd_star(*args):
    gcd = 0
    for a in args:
        gcd = math.gcd(gcd, a)
    return gcd

def downsample(lt: LatticeTensor, s_matrix: torch.IntTensor):

    with torch.no_grad():
        B = torch.stack(lt.parent.coset_vectors, dim=-1)
        B = torch.cat([B, torch.diag(lt.parent.coset_scale)], dim=-1).numpy()
        L, U = column_style_hermite_normal_form(B)
        D = s_matrix.numpy()
        s = lt.parent.dimension
        L = D @ L[:s, :s]
        D, c = find_coset_vectors(L)
        coset_stride = [int(_.detach().item()) for _ in D.diag()//lt.parent.coset_scale]

    coset_data = []
    for v in c:
        if all([_ == 0 for _ in v]):
            slc = [slice(0, None), slice(0, None)] + [slice(0, None, int(cs)) for cs in coset_stride]
            data = lt.coset(0).__getitem__(slc)
            coset_data += [(v, data)]
        else:

            # Since we subsampled the lattice M, we know that MS' basis is in
            # the original lattice
            coset_index = lt.parent.coset_index(v)
            vector = lt.parent.coset_offset(coset_index)

            # Now we shift it until we find the "left-most" (according to each dimension)
            # coordinate for this coset
            for _ in range(s):
                w = -D @ torch.IntTensor([1 if __ == _ else 0 for __ in range(s)])

                if lt.on_lattice(vector + w):
                    vector = vector + w

            array_index, coset_start = lt.raw_coset_point_info(vector)
            slc = [slice(0, None), slice(0, None)] + [slice(0, None, int(c)) for s, c in zip(coset_start, coset_stride)]
            data = lt.coset(array_index).__getitem__(slc)
            coset_data += [(v, data)]

    # Now remove the gcd
    gcd1 = _gcd_star(*D.diag())
    gcd2 = _gcd_star(*sum([list(v) for v, _ in coset_data], []))
    gcd = math.gcd(gcd1, gcd2)


    sl = Lattice([torch.IntTensor([c/gcd for c in v]) for v, _ in coset_data], D.diag()/gcd)

    return sl(
        {
            v: data for v, data in coset_data
        }
    )

def upsample(lt: LatticeTensor, s_matrix: torch.IntTensor):
    D = np.array([[1, -1], [1, 1]])
    k = np.linalg.det(D)
    np.linalg.inv(D) * k


    with torch.no_grad():
        B = torch.stack(lt.parent.coset_vectors, dim=-1)
        B = torch.cat([B, torch.diag(lt.parent.coset_scale)], dim=-1).numpy()
        L, U = column_style_hermite_normal_form(B)
        D = s_matrix.numpy()
        s = lt.parent.dimension
        L = D @ L[:s, :s]
        D, c = find_coset_vectors(L)
        coset_stride = [int(_.detach().item()) for _ in D.diag()//lt.parent.coset_scale]

    coset_data = []
    for v in c:
        if all([_ == 0 for _ in v]):
            slc = [slice(0, None), slice(0, None)] + [slice(0, None, int(cs)) for cs in coset_stride]
            data = lt.coset(0).__getitem__(slc)
            coset_data += [(v, data)]
        else:

            # Since we subsampled the lattice M, we know that MS' basis is in
            # the original lattice
            coset_index = lt.parent.coset_index(v)
            vector = lt.parent.coset_offset(coset_index)

            # Now we shift it until we find the "left-most" (according to each dimension)
            # coordinate for this coset
            for _ in range(s):
                w = -D @ torch.IntTensor([1 if __ == _ else 0 for __ in range(s)])

                if lt.on_lattice(vector + w):
                    vector = vector + w

            array_index, coset_start = lt.raw_coset_point_info(vector)
            slc = [slice(0, None), slice(0, None)] + [slice(0, None, int(c)) for s, c in zip(coset_start, coset_stride)]
            data = lt.coset(array_index).__getitem__(slc)
            coset_data += [(v, data)]

    # Now remove the gcd
    gcd1 = _gcd_star(*D.diag())
    gcd2 = _gcd_star(*sum([list(v) for v, _ in coset_data], []))
    gcd = math.gcd(gcd1, gcd2)


    sl = Lattice([torch.IntTensor([c/gcd for c in v]) for v, _ in coset_data], D.diag()/gcd)

    return sl(
        {
            v: data for v, data in coset_data
        }
    )