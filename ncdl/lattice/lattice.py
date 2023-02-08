"""
This is largely a rework of the ideas within some of the open research I did at Huawei
in 2021. Most of the ideas in that work we're not well tested, and some of the implementation
was a little sloppy (this is all on me).

"""

from ncdl.utils import get_coset_vector_from_name, interval_differences
from typing import List, Tuple, Union
from collections.abc import Iterable
import numpy as np
import torch


class LatticeTensor:
    """
    A LatticeTensor container is the base data structure for processing data on non-Cartesian lattices. It extends the
    concept of tensor to
    """

    def __init__(self,
                 lt: "LatticeTensor" = None,
                 alt_cosets: List[torch.Tensor] = None,
                 parent: "Lattice" = None,
                 alt_offsets: List[torch.IntTensor] = None):
        """
        Generally, an end-user should not be using this constructor. This constructor is meant to be used by



        """

        # If we have an alternative set of cosets we want to construct
        # ensure that they have the same shape
        if alt_cosets is not None and lt is not None:
            assert all([a.size() == b.size() for (a, b) in zip(lt._cosets, alt_cosets)])

        if parent is None and lt is None:
            raise ValueError("This lattice tensor has no viable parent?")

        self.eps = 1e-6
        self.parent = None
        if lt is not None:
            self.parent = lt.parent

        if parent is not None:
            self.parent = parent

        coset_offsets = parent._coset_vectors[:] if alt_offsets is None else alt_offsets
        coset_offsets = [np.array(_, dtype='int') for _ in coset_offsets]
        if alt_cosets is not None:
            cosets = alt_cosets
        elif lt is not None:
            cosets = [_.detach().clone() for _ in lt._cosets]
        else:
            raise ValueError(f"No lattice data for constructor!")

        self._coset_offsets, self._cosets = self.parent._coset_sort(coset_offsets, cosets)
        self._coset_offsets = [torch.IntTensor([int(e) for e in csv]) for csv in self._coset_offsets]
        self._validate()

    def _validate(self):
        """
        This function validates that this lattice tensor is properly formated -- i.e. consistent
        """
        # check all batch sizes
        bs = self._cosets[0].shape[0]
        if any([c.shape[0] != bs for c in self._cosets]):
            raise ValueError(f"Batch size mismatch among child tensors in lattice tensor")

        if any([len(c.shape) != self.parent.dimension + 2 for c in self._cosets]):
            raise ValueError(f"Invalid tensor dimension input")

        if len(self._cosets) == 1:
            return True

        # check that the coset overlap correctly
        for axis in range(self.parent.dimension):
            boundaries = []
            for coset_vector, coset in zip(self._coset_offsets, self._cosets):
                start, end = self._project_axis_bounds(coset_vector, self.parent.coset_scale, coset.shape[2 + axis],
                                                       axis)
                boundaries += [(int(start.item()), int(end.item()))]
            intervals = interval_differences(boundaries)
            for begin, end in intervals:
                if abs(begin - end) >= self.parent.coset_scale[axis]:
                    raise ValueError(f"Tensors in lattice tensor don't properly interleave!")
        return True

    def detach(self) -> "LatticeTensor":
        return LatticeTensor(self, alt_cosets=[_.detach() for _ in self._cosets])

    def clone(self) -> "LatticeTensor":
        return LatticeTensor(self, alt_cosets=[_.clone() for _ in self._cosets])

    def shift_constants(self, i, j, deconv=False):
        w = -1 if deconv else 1
        kappa_shift = np.array(self.coset_vectors[self.parent.kappa(i, j)])
        shift = w*(self._coset_offsets[j] - self._coset_offsets[i]) - kappa_shift
        shift = (1/self.parent.coset_scale) * shift
        return np.array(shift.round()).astype('int')

    def delta(self, i):
        delta = np.array(self._coset_offsets[i], dtype='int')
        delta = delta - (delta % np.array(self.parent.coset_scale, dtype='int'))
        delta = (1/self.parent.coset_scale) * delta
        return np.array(delta.round(), dtype='int')

    @property
    def device(self):
        return self.coset(0).device

    def to(self, device) -> "LatticeTensor":
        return LatticeTensor(self,
                             alt_cosets=[_.to(device) for _ in self._cosets],
                             alt_offsets=self._coset_offsets)

    def is_tensor(self) -> bool:
        return len(self._cosets) == 1

    def __cmp__(self, other):
        raise NotImplementedError()

    def coset(self, coset: int) -> torch.Tensor:
        if not (0 <= coset < len(self._cosets)):
            raise ValueError(f"Invalid coset index {coset}")
        return self._cosets[coset]

    def lattice_bounds(self):
        """
        Returns the inclusive boundary of the lattice tensor.

        This is perhaps a little complicated, so I'll clarify this with an example
          |o   o   o
         x|  x   x
          |o   o   o
         x|  x   x
          |o---o---o
         x   x   x

        The example above should be (0, batch_size-1)

        """
        batch_size = self._cosets[0].shape[0]
        channel_size = self._cosets[0].shape[1]
        mins = [0] * self.parent.dimension
        maxs = None
        with torch.no_grad():
            for idx, c in enumerate(self._cosets):
                # Get the raw array dimension, then scale it by the coset scale
                dimensions = [(array_dim-1) * dim_scale.item() for array_dim, dim_scale in zip(c.shape[2:], self.parent.coset_scale)]

                # Offset that by the coset offest
                dimensions = [array_dim + dim_offset.item() for array_dim, dim_offset in zip(dimensions, self._coset_offsets[idx])]

                if maxs is None:
                    maxs = dimensions
                else:
                    maxs = [int(max(a, _)) for a, _ in zip(dimensions, maxs)]

                mm = [int(z.item()) for z in self._coset_offsets[idx]]
                mins = [int(min(a, _)) for a, _ in zip(mm, mins)]
        return (0, batch_size-1), (0, channel_size-1), *list(zip(mins, maxs))

    def on_lattice(self, p : torch.IntTensor):
        b, c, *dim = self.lattice_bounds()
        if self.parent.coset_index(p) is not None:
            if all([d[0] <= _ <= d[1] for (_, d) in zip(p, dim)]):
                return True
        return False

    def __getitem__(self, slices) -> "LatticeTensor":
        """
        lt[batch slice, rectangular region in space of integers] -> LatticeTensor
        """
        raise NotImplementedError
        batch_slice = slices[0]
        channel_slice = slices[1]

        tensor_slices = slices[2:]
        _, _, *tensor_bounds = self.lattice_bounds()

        # Check that the tensor slices are within the lattice tensor's boundary
        start_pt = []
        end_pt = []
        for slc, bnd in zip(tensor_slices, tensor_bounds):
            start = slc.start
            stop = slc.stop

            if start is None:
                start = 0

            if stop is not None and stop < 0:
                stop = bnd[-1] - stop - 2

            if stop is None:
                stop = bnd[-1] - 1

            if start not in range(*bnd) or stop not in range(*bnd):
                raise ValueError(f"Lattice tensor slice out of bound")

            if slc.step != 1 and slc.step is not None:
                raise ValueError(f"Step must be 1 for tensor slices")

            start_pt += [start]
            end_pt += [stop]

        if self.parent.coset_index(start_pt) is not None:
            return 0
        else:
            raise NotImplementedError()

    def _find_correspondence(self, other: "LatticeTensor"):
        """

        """
        if len(self._coset_offsets) != len(other._coset_offsets):
            raise ValueError("LatticeTensors must belong to the same lattice")
        with torch.no_grad():
            c_a = sum(self._coset_offsets) / len(self._coset_offsets)
            c_b = sum(other._coset_offsets) / len(other._coset_offsets)

            a = [_ - c_a for _ in self._coset_offsets]
            b = [_ - c_b for _ in other._coset_offsets]

            corresp = [0] * len(a)
            found = 0
            for i, p in enumerate(a):
                for j, q in enumerate(b):
                    if (p - q).sum().abs().item() < self.eps:
                        if tuple(self._cosets[i].shape) == tuple(other._cosets[j].shape):
                            corresp[i] = j
                            found += 1
                            break
            if found != len(a):
                raise ValueError('Lattices do not have a correspondence between cosets')
        return corresp

    def __add__(self, other):
        correspondence = self._find_correspondence(other)
        keys = {}
        with torch.no_grad():
            for i, (offset, coset_a) in enumerate(zip(self._coset_offsets, self._cosets)):
                offset = tuple([int(_) for _ in offset])
                keys[tuple(offset)] = coset_a + other._cosets[correspondence[i]]
        return self.parent(keys)

    def __sub__(self, other):
        correspondence = self._find_correspondence(other)
        keys = {}
        with torch.no_grad():
            for i, (offset, coset_a) in enumerate(zip(self._coset_offsets, self._cosets)):
                offset = tuple([int(_) for _ in offset])
                keys[tuple(offset)] = coset_a - other._cosets[correspondence[i]]
        return self.parent(keys)

    def __mul__(self, other):
        correspondence = self._find_correspondence(other)
        keys = {}
        with torch.no_grad():
            for i, (offset, coset_a) in enumerate(zip(self._coset_offsets, self._cosets)):
                offset = tuple([int(_) for _ in offset])
                keys[tuple(offset)] = coset_a * other._cosets[correspondence[i]]
        return self.parent(keys)

    def raw_coset_point_info(self, lattice_site):
        coset = self.parent.coset_index(lattice_site)
        if coset is None:
            return None
        return coset, (lattice_site - self.coset_vector(coset))//self.parent.coset_scale

    @staticmethod
    def _project_axis_bounds(coset_vector, coset_scale, element_count, axis):
        start = coset_vector[axis]
        end = coset_scale[axis] * element_count + coset_vector[axis]
        return start, end

    @property
    def coset_vectors(self):
        return  self._coset_offsets[:]

    def coset_vector(self, coset: int) -> torch.IntTensor:
        return self._coset_offsets[coset]


class Lattice:
    """
    The general "LatticeTensor" factory. Basically, this holds all the important information about the
    point structure of the Lattice. LatticeTensors are instances of multi-dimensional sequences, but they
    store values only within a bounded region (and on an integer lattice).
    """

    def __init__(self,
                 input_lattice: Union[List, str],
                 scale: Union[torch.IntTensor, None] = None,
                 tensor_backend: torch._tensor = torch.Tensor):
        """
        Instantiate the LatticeTensor factory.

        :param input_lattice: Either a lattice name (i.e. quincunx, qc, bcc etc..) or a list of coset vectors.
        :param scale: If input lattice is a list of coset vectors, then this should be an n-dimensional integer vector
                      specifying the scale of each coset.
        :param tensor_backend: The type of tensor that backs this tensor created from this factory. This will mostly be
                                torch.Tensor, but can be any torch._tensor type.
        """
        coset = input_lattice
        if isinstance(input_lattice, str):
            if scale is not None:
                print("Warning: Scale was provided to a pre-configured lattice, it will be ignored.")
            coset, scale = get_coset_vector_from_name(input_lattice)

        if not isinstance(coset, list):
            raise ValueError(f"The input 'input_lattice' should either be a common lattice identifier "
                             f"or a list of IntTensors on the CPU that represent the coset structure "
                             f"of the lattice. If you specified the latter case, then you should also "
                             f"pass 'scale' as matrix (torch.IntTensor) that describes the scale of the "
                             f"cosets.")

        if len(coset) == 0:
            raise ValueError(f"The input 'coset' should be a non-empty list of IntTensors")

        # TODO: Change this to np.array
        if any([not isinstance(_, torch.IntTensor) for _ in coset]):
            raise ValueError(f"The input 'coset' should be a non-empty list of IntTensors")

        if any([_.size() != coset[0].size() for _ in coset]):
            raise ValueError(f"The input 'coset' should be a non-empty list of 1-D IntTensors with the same size")

        self.tensor = tensor_backend
        self._dimension = coset[0].shape[0]
        self._coset_scale = scale

        with torch.no_grad():
            canonical_offsets = [_ % scale.abs() for _ in coset]
            # TODO: change to np.array(, dtype='int')
            self._coset_vectors = sorted(canonical_offsets,
                                         key=lambda x: sum([abs(_.item()) for _ in x[:]])
            )

    @property
    def coset_vectors(self):
        return self._coset_vectors[:]

    @property
    def dimension(self) -> int:
        return self._dimension

    @property
    def coset_scale(self) -> torch.IntTensor:
        return self._coset_scale.detach()

    @property
    def coset_count(self) -> int:
        return len(self._coset_vectors)

    def coset_index(self, pt: Union[Tuple[int], List[int], torch.IntTensor]) -> Union[int, None]:
        return self.cartesian_index(pt)[0]

    def cartesian_index(self, pt: Union[Tuple[int], List[int], torch.IntTensor]):
        # TODO: change to np.array(, dtype='int')
        if type(pt) == tuple or type(pt) == list:
            pt = torch.IntTensor([int(_) for _ in pt])
        for idx, c in enumerate(self._coset_vectors):
            x = (1 / self._coset_scale) * (pt - c)
            if all([(_.round() - _).abs() < 2 * torch.finfo().eps for _ in x]):
                return idx, x
        return None, None

    def coset_offset(self, idx):
        # TODO: change to np.array(, dtype='int')
        if 0 > idx <= self.coset_count:
            raise ValueError('Invalid coset offset index!')
        return self._coset_vectors[idx].detach()

    def cartesian_to_lattice(self, pt, coset_index):

        return self._coset_scale * torch.tensor(pt) + self._coset_vectors[coset_index]

    def __cmp__(self, other):
        # TODO: Check if tensors implement a cmp method, then implement this
        raise NotImplementedError()

    def _validate_coset_vectors(self, vectors):
        #
        pass

    def __eq__(self, other):
        eq = all([(a - b).abs().sum() < 1e-5 for a,b in zip(self.coset_vectors, other.coset_vectors)])
        return eq and all([(a - b).abs().sum() < 1e-5 for a,b in zip(self.coset_scale, other.coset_scale)])

    def _coset_sort(self, vectors, cosets=None):
        ret_cosets = True

        if cosets is None:
            ret_cosets = False
            cosets = [None] * len(vectors)

        if len(vectors) != self.coset_count:
            raise ValueError("invalid number of cosets!")

        to_sort = zip(vectors, cosets)
        vectors, cosets = zip(*sorted(to_sort, key=lambda x: (sum([abs(_).item() for _ in x[0]]))))
        if ret_cosets:
            return list(vectors), list(cosets)
        return list(vectors)

    def __call__(self, *args):

        vectors = self._coset_vectors[:]

        if isinstance(args[0], dict):
            if len(args) > 1:
                raise ValueError("Too many arguments!")

            with torch.no_grad():
                vectors = [torch.IntTensor(_) for _ in args[0]]
                self._validate_coset_vectors(vectors)
                cosets = [args[0][_] for _ in args[0]]
                vectors, cosets = self._coset_sort(vectors, cosets)

        else:
            cosets = args
        device = cosets[0].device
        if any([device != _.device for _ in cosets]):
            raise ValueError("Not all cosets on same device")

        if len(cosets) != self.coset_count:
            raise ValueError(f"Invalid number of input tenors, expected {self.coset_count}, but got {len(cosets)}")

        if any([not isinstance(_, self.tensor) for _ in cosets]):
            raise ValueError(f"All input tensors must be of type {self.tensor}")

        return LatticeTensor(None, cosets, parent=self, alt_offsets=vectors)

    def kappa(self, i, j, additive=False) -> bool:
        r = tuple([a-b if not additive else a+b for a,b in zip(self._coset_vectors[j], self._coset_vectors[i])])
        return self.coset_index(r)


class HalfLattice(Lattice):
    def __init__(self,
                 input_lattice: Union[List, str],
                 scale: Union[torch.FloatTensor, None] = None):
        super(HalfLattice, self).__init__(input_lattice, scale, torch.HalfTensor)


class FloatLattice(Lattice):
    def __init__(self,
                 input_lattice: Union[List, str],
                 scale: Union[torch.FloatTensor, None] = None):
        super(FloatLattice, self).__init__(input_lattice, scale, torch.FloatTensor)


class DoubleLattice(Lattice):
    def __init__(self,
                 input_lattice: Union[List, str],
                 scale: Union[torch.FloatTensor, None] = None):
        super(DoubleLattice, self).__init__(input_lattice, scale, torch.DoubleTensor)


class IntLattice(Lattice):
    def __init__(self,
                 input_lattice: Union[List, str],
                 scale: Union[torch.FloatTensor, None] = None):
        super(IntLattice, self).__init__(input_lattice, scale, torch.IntTensor)
