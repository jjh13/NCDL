from typing import Tuple, List
import torch


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

    raise ValueError(f"Lattice '{name}' is not valid?")
