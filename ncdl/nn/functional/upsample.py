from itertools import product
from hsnf import column_style_hermite_normal_form
from ncdl.lattice import Lattice, LatticeTensor
import numpy as np
import torch
import math
from ncdl.utils import find_coset_vectors, _gcd_star

def _in_bounds(bounds, ls):
    bounds = bounds[2:]
    for i, (start, end) in enumerate(bounds):
        if not (start <= ls[i] <= end):
            return False
    return True

def upsample_lattice(lattice: Lattice, s_matrix: np.array):
    k = np.abs(np.linalg.det(s_matrix))
    D = np.linalg.inv(s_matrix)

    B = np.stack(lattice.coset_vectors, axis=-1)
    B = np.concatenate([B, np.diag(lattice.coset_scale)], axis=-1)
    L, U = column_style_hermite_normal_form(k*B)
    s = lattice.dimension
    L = D @ L[:s, :s]
    D, c = find_coset_vectors(L)

    gcd1 = _gcd_star(*D.diagonal())
    gcd2 = _gcd_star(*sum([list(v) for v in c], []))
    gcd = math.gcd(gcd1, gcd2)

    scale = D.diagonal()//gcd

    return Lattice([np.array([_/gcd for _ in v], dtype='int') for v in c], np.array(scale, dtype='int'))



def upsample(lt: LatticeTensor, s_matrix: torch.IntTensor):
    k = np.abs(np.linalg.det(s_matrix))
    D = np.linalg.inv(s_matrix)

    # Get the boundary of the scaled input
    bounds = [(k * _[0], k * _[1]) for _ in lt.lattice_bounds()]

    with torch.no_grad():
        B = np.stack(lt.parent.coset_vectors, axis=-1)
        B = np.concatenate([B, np.diag(lt.parent.coset_scale)], axis=-1)
        L, U = column_style_hermite_normal_form(k*B)
        s = lt.parent.dimension
        L = D @ L[:s, :s]
        D, c = find_coset_vectors(L)
        stride = k * lt.parent.coset_scale // D.diagonal()
        stride = [int(_) for _ in stride]

    batch, channels = lt.coset(0).shape[:2]

    cosets = []

    # now we push all cosets as far left as we can, and allocate the array
    for v in c:

        v = np.array(v, dtype='int')
        # Push Left
        for _ in range(s):
            w = -D @ np.array([1 if __ == _ else 0 for __ in range(s)], dtype='int')
            if _in_bounds(bounds, v + w):
                v = v + w

        # Create an all zero array for the coset
        local_size = [ int((end - v[i])/D.diagonal()[i]) + 1 for (i,(start, end)) in enumerate(bounds[2:])]
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
                array_index = (offset - v_prime)/D.diagonal()
                if np.abs(np.modf(array_index)[0]).sum() > 1e-4:
                    continue
                # if torch.frac(array_index).abs().sum().item() > 1e-4:
                #     continue
                array_index = [int(_) for _ in array_index]

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
    cosets = [(v - o_shift, data) for v, data in cosets]

    # Now remove the gcd
    gcd1 = _gcd_star(*D.diagonal())
    gcd2 = _gcd_star(*sum([list(v) for v, _ in cosets], []))
    gcd = math.gcd(gcd1, gcd2)
    cosets = [(v//gcd, data) for v,data in cosets]

    # Finally construct the lattice object
    sl = Lattice([np.array([c//gcd for c in v], dtype='int') for v, _ in cosets], D.diagonal()/gcd)

    return sl(
        {
            tuple([int(_) for _ in v]): data for v, data in cosets
        }
    )