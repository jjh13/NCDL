import torch
from hsnf import column_style_hermite_normal_form
from ncdl.lattice import Lattice, LatticeTensor
import numpy as np


from itertools import product
import math
from ncdl.nn.functional.upsample import find_coset_vectors, _gcd_star

def downsample_lattice(lattice: Lattice, s_matrix: np.array):
    B = np.stack(lattice.coset_vectors, axis=-1)
    B = np.concatenate([B, torch.diag(lattice.coset_scale)], axis=-1)
    L, U = column_style_hermite_normal_form(B)
    D = s_matrix.numpy()
    s = lattice.dimension
    L = D @ L[:s, :s]
    D, c = find_coset_vectors(L)

    gcd1 = _gcd_star(*D.diagonal())
    gcd2 = _gcd_star(*sum([list(v) for v in c], []))
    gcd = math.gcd(gcd1, gcd2)

    scale = D.diagonal()//gcd

    return Lattice([torch.IntTensor([_/gcd for _ in v]) for v in c], torch.IntTensor(scale))


def downsample(lt: LatticeTensor, s_matrix: torch.IntTensor):

    with torch.no_grad():
        B = torch.stack(lt.parent.coset_vectors, dim=-1)
        B = torch.cat([B, torch.diag(lt.parent.coset_scale)], dim=-1).numpy()
        L, U = column_style_hermite_normal_form(B)
        D = s_matrix.numpy()
        s = lt.parent.dimension
        L = D @ L[:s, :s]
        D, c = find_coset_vectors(L)
        D = torch.IntTensor(D)
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