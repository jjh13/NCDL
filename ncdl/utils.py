from typing import Tuple, List
import torch


def get_coset_vector_from_name(name: str) -> Tuple[List[torch.IntTensor], torch.Tensor]:
    """
    For a canonical

    :param name:
    :return: a tuple of torch tensor
    """
    raise NotImplementedError(name)
    # return [torch.IntTensor([1, 1], requires_grad=False)], torch.tensor()
