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

def _in_bounds(bounds, ls):
    bounds = bounds[2:]
    for i, (start, end) in enumerate(bounds):
        if not (start <= ls[i] <= end):
            return False
    return True

def upsample(lt: LatticeTensor, s_matrix: torch.IntTensor):
    k = np.abs(np.linalg.det(s_matrix))
    D = np.linalg.inv(s_matrix)

    # Get the boundary of the scaled input
    bounds = [(k * _[0], k * _[1]) for _ in lt.lattice_bounds()]

    with torch.no_grad():
        B = torch.stack(lt.parent.coset_vectors, dim=-1)
        B = torch.cat([B, torch.diag(lt.parent.coset_scale)], dim=-1).numpy()
        L, U = column_style_hermite_normal_form(k*B)
        s = lt.parent.dimension
        L = D @ L[:s, :s]
        D, c = find_coset_vectors(L)
        stride = k * lt.parent.coset_scale // D.diag()
        stride = [int(_.item()) for _ in stride]

    batch, channels = lt.coset(0).shape[:2]

    cosets = []

    # now we push all cosets as far left as we can, and allocate the array
    for v in c:

        v = torch.IntTensor(v)
        # Push Left
        for _ in range(s):
            w = -D @ torch.IntTensor([1 if __ == _ else 0 for __ in range(s)])
            if _in_bounds(bounds, v + w):
                v = v + w

        # Create an all zero array for the coset
        local_size = [ int((end - v[i])/D.diag()[i]) + 1 for (i,(start, end)) in enumerate(bounds[2:])]
        coset = torch.zeros((batch, channels, *local_size), device=lt.device)
        cosets += [(v, coset)]

    # Next, we copy all of the data to the new arrays
    for idx in range(lt.parent.coset_count):
        # Get our offset and data
        offset = lt.coset_vector(idx) * k
        data = lt.coset(idx)
        injected = False

        # Search for the coset we're going to inject into
        for v_prime, dest_coset in cosets:
            with torch.no_grad():
                array_index = (offset - v_prime)/D.diag()
                if torch.frac(array_index).abs().sum().item() > 1e-4:
                    continue
                array_index = [int(_.item()) for _ in array_index]

                # Construct the slices for the assignment, we start with the default slice that
                # selects all batches and channels
                coset_slice = [
                    slice(0, None, 1),
                    slice(0, None, 1)
                ] + [slice(a_idx, None, stride[iidx]) for iidx, a_idx in enumerate(array_index)]
            dest_coset[coset_slice] = data
            injected = True
            break

        if not injected:
            raise ValueError("??")

    # The penultimate step is to remove any shift from the first coset, then
    # remove the GCD to keep the lattice structures from getting too large -- this isn't a technical problem
    # but I'd like to avoid this creep, just to keep our integers in reasonable ranges without overflowing
    o_shift, _ = cosets[0]
    cosets = [(v - o_shift, data) for v,data in cosets]

    # Now remove the gcd
    gcd1 = _gcd_star(*D.diag())
    gcd2 = _gcd_star(*sum([list(v) for v, _ in cosets], []))
    gcd = math.gcd(gcd1, gcd2)
    cosets = [(v//gcd, data)for v,data in cosets]

    # Finally construct the lattice object
    sl = Lattice([torch.IntTensor([c//gcd for c in v]) for v, _ in cosets], D.diag()/gcd)

    return sl(
        {
            v: data for v, data in cosets
        }
    )