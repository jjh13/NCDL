from ncdl.lattice import Lattice, LatticeTensor
from typing import List, Tuple, Union
from collections import defaultdict
import torch.nn.functional as F
import itertools


def padding_for_stencil(l: Union[Lattice, LatticeTensor],
                        filter_stencil: Union[List[List[int]]],
                        center: Union[List[int], Tuple[int]]):
    # if center not in filter_stencil:
    #     raise ValueError("The stencil must contain the origin. ")
    filter_stencil = filter_stencil[:]

    if isinstance(l, LatticeTensor):
        l = l.parent
    if not isinstance(l, Lattice):
        raise ValueError("Input must be a Lattice or LatticeTensor object")

    # Add the zero element to the stencil -- it needs to be aware
    # of this point, or the following procedure will underestimate
    # the padding. This is the simplest way to do it
    filter_stencil += [tuple([0 for _ in range(l.dimension)])]
    filter_stencil = [[a - b for (a, b) in zip(pt, center)] for pt in filter_stencil]

    maxs = [max(_) for _ in zip(*filter_stencil)]
    mins = [min(_) for _ in zip(*filter_stencil)]

    total_coverage = []
    for ls in itertools.product(*[range(a, b + 1) for a, b in zip(mins, maxs)]):
        if l.coset_index(ls) is None:
            continue
        total_coverage += [ls]

    padding = []
    for _ in range(l.dimension):
        left, right = 0, 0
        proj = {p[_]: _ for p in total_coverage}
        for p in proj.keys():
            if p < 0:
                left += 1
            if p > 0:
                right += 1
        padding += [[left, right]]
    return sum(reversed(padding), [])


def pad(lt: LatticeTensor, pad, mode='constant', value=0.0):
    """
    Pads the given lattice tensor using lattice padding.

    :param lt: The input LatticeTensor to pad.
    :param pad: The amount of padding along each dimension.
    :param mode: The mode of padding -- see torch.nn.functional.pad for details.
    :param value: The value to fill the padding with if in constant mode.
    :return: LatticeTensor with padding.
    """
    parent = lt.parent
    dim = parent.dimension
    num_cosets = parent.coset_count
    scale = parent.coset_scale
    if mode == 'zero':
        mode = 'constant'
        value = 0.0

    assert len(pad) % 2 == 0
    assert len(pad) // 2 <= dim

    offsets = lt.coset_vectors
    parent_offsets = parent.coset_vectors
    coset_padding = [[[0, 0] for __ in range(dim)] for _ in range(num_cosets)]

    # Iterate over the pairs of padding
    for d, (left_pad, right_pad) in enumerate(zip(pad[::2], pad[1::2])):
        d = dim - 1 - d
        proj_offsets = [(coset_id, offset[d]) for coset_id, offset in enumerate(parent_offsets)]

        # Bin together any cosets that have the same projected offsets
        bins = defaultdict(list)
        for coset_id, offset in proj_offsets:
            bins[offset] += [coset_id]

        # If the padding is a multiple of the coset width, then we can safely fully and equally pad each coset
        proj_offsets = sorted(bins.keys())
        uniform_padding_size = len(proj_offsets)
        full_right_pad = right_pad // uniform_padding_size
        full_left_pad = left_pad // uniform_padding_size
        for coset in range(num_cosets):
            coset_padding[coset][dim - d - 1] = [full_left_pad, full_right_pad]

        # We might have some remaining padding
        dscale = int(scale[d].item())
        part_right_pad = right_pad % uniform_padding_size
        part_left_pad = left_pad % uniform_padding_size

        # If we're padding on the right, then we don't have to mess with the offsets vector at all
        if part_right_pad > 0:
            logical_cs = max(range(num_cosets), key=lambda i: lt.coset(i).shape[d + 2] * dscale + offsets[i][d])

            for _, poffset in zip(range(part_right_pad), proj_offsets[1:]):
                for cs in bins[poffset]:
                    cs_to_pad = parent.kappa(logical_cs, cs, True)
                    coset_padding[cs_to_pad][dim - d - 1][1] += 1

        if part_left_pad > 0:
            # if we're padding on the left, then we project the lt's
            # current coset offsets
            proj = [(cid, offset[d]) for cid, offset in enumerate(offsets)]
            bins = defaultdict(list)
            for coset_id, offset in proj:
                bins[offset] += [coset_id]

            # Sort the unique projected offsets
            proj_offsets = sorted(list(bins.keys()), reverse=True)
            for _, poffset in zip(range(part_left_pad), proj_offsets):
                for cs in bins[poffset]:
                    cs_to_pad = cs

                    # if we're padding the 0'th coset, then we can reset
                    # all the coset offsets to their default value in this dim
                    if cs_to_pad == 0:
                        for _ in range(1, num_cosets):
                            offsets[_] = tuple([a - int(b) for a, b in zip(offsets[_],
                                                                           [-dscale if _ == d else 0 for _ in
                                                                            range(dim)])])
                    else:
                        offsets[cs_to_pad] = tuple([a - int(b) for a, b in zip(offsets[cs_to_pad],
                                                                               [dscale if _ == d else 0 for _ in
                                                                                range(dim)])])

                    coset_padding[cs_to_pad][dim - d - 1][0] += 1

    # pad each coset
    cs = [F.pad(lt.coset(_), pad=sum(coset_padding[_], []), mode=mode, value=value) for _ in range(num_cosets)]

    # Construct a new lattice tensor object
    new_lt = LatticeTensor(None, parent=lt.parent, alt_cosets=cs, alt_offsets=offsets)
    return new_lt

def pad_like_lattice_tensor(lt: LatticeTensor, pad, mode='constant', value=0.0):

    # Check that
    pass