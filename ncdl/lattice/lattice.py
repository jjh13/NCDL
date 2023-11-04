"""

"""

from ncdl.utils import get_coset_vector_from_name, interval_differences
from typing import List, Tuple, Union, Dict, Optional
import numpy as np
import torch


class LatticeTensor:
    """
    A LatticeTensor container is the base data structure for processing data on
    non-Cartesian lattices. It extends the concept of the tensor to non-square
    grids. Note that this is a base class, it need not inherent form any
    PyTorch base clasee.
    """

    def __init__(self,
                 lt: "LatticeTensor" = None,
                 alt_cosets: List[torch.Tensor] = None,
                 parent: "Lattice" = None,
                 alt_offsets: List[np.array] = None):
        """
        Generally, an end-user should not be using this constructor. This
        constructor is meant to be used by by the Lattice factory.

        Note that this constructor is VERY bare-bones. It doesn't enforce the
        ordering principle mentioned in the paper (the Lattice factory does
        that).

        :param lt: An input "protoypical" lattice tensor. If all other params
                   are None this will constructor will simply clone this data.
                   Otherwise, this forms the base for the region (and
                   therefore coset vectors) of the lattice tensor.

        :param alt_cosets: When alt_cosets is specified, lt must be specified.
                   this will clone the structure of lt, but directly use the
                   data in the tensors specified in alt_cosets.

        :param parent: When parent is specified, then this simply sets the
                   parent of the resultant lattice tensor to the appropriate
                   Lattice factory. If lt is specified, we will also check
                   against that lattice tensor's parent

        :param alt_offsets:
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
        self._coset_offsets = [np.array([int(e) for e in csv], dtype='int') for csv in self._coset_offsets]
        self._validate()

    def _validate(self):
        """
        This function validates that this lattice tensor is properly formated.
        I.e. consistent (interleaves, channels agree, etc...)
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
        """
        Analog of tensor.detach(), returns a new LatticeTensor that is the same
        as the input, but detached from the computational graph.

        :return: Detached lattice tensor.
        """
        return LatticeTensor(self, alt_cosets=[_.detach() for _ in self._cosets])

    def clone(self) -> "LatticeTensor":
        """
        Analog of tensor.clone(), returns a new LatticeTensor copy that is
        still part of the computational graph.

        :return: Cloned lattice tensor.
        """
        return LatticeTensor(self, alt_cosets=[_.clone() for _ in self._cosets])

    def shift_constants(self, i: int, j: int, deconv: bool = False) -> np.array:
        """
        This is the analogue of \delta(i,j) in the paper. Its a shift that is
        used multiple different operations. See the appropriate proposition
        where \delta(i,j) is defined in the paper.

        :param i: Coset i
        :param j: Coset j
        :param deconv: Is this funciton used for deconvolution?

        :return: an integer np.array containing the shift.
        """
        w = -1 if deconv else 1
        kappa_shift = np.array(self.coset_vectors[self.parent.kappa(i, j)])
        shift = w*(self._coset_offsets[j] - self._coset_offsets[i]) - kappa_shift
        shift = (1/self.parent.coset_scale) * shift
        return np.array(shift.round()).astype('int')

    @property
    def device(self) -> torch.device:
        """
        Returns the current device that the lattice tensor resides on.

        Assumes that the lattice tensor is consistent (i.e. the tensors
        underlying the LT have not been erroneously moved)

        :return: A torch device.
        """
        return self.coset(0).device

    def to(self, device: torch.device) -> "LatticeTensor":
        """
        Moves the lattice tensor to an appropriate device.

        :param device: A valid torch.device instance.

        :return: A lattice tensor instance on device.
        """
        return LatticeTensor(self,
                             alt_cosets=[_.to(device) for _ in self._cosets],
                             alt_offsets=self._coset_offsets)

    def is_tensor(self) -> bool:
        """
        Returns true if the current lattice tensor is a Cartesian lattice tensor.

        :return: Boolean value indicating Cartesian-ness.
        """
        return len(self._cosets) == 1


    def coset(self, coset: int) -> torch.Tensor:
        """
        Returns the underlying tensor for a given coset index. Gradients are
        tracked for these tensors.

        :return: A torch tensor on self.device
        """
        if not (0 <= coset < len(self._cosets)):
            raise ValueError(f"Invalid coset index {coset}")
        return self._cosets[coset]

    @property
    def coset_vectors(self) -> List[np.array]:
        """
        A list of all the coset vectors for this lattice tensor. Equivalent to
        the $v_^\mathcal{R}_i$ in the paper.
        """
        return self._coset_offsets[:]

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

        The example above should return (0, batch_size-1), (0, channels - 1), (-1, 2), (-1, 2)

        """
        batch_size = self._cosets[0].shape[0]
        channel_size = self._cosets[0].shape[1]
        mins = [0] * self.parent.dimension
        maxs = None
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

    def on_lattice(self, p: np.array) -> bool:
        """
        Tests if a point, represented as an integer numpy array is on the given
        lattice.

        :param p: an integer numpy array with dimension s.
        :return: True if the point is on a lattice, False otherwise.
        """
        b, c, *dim = self.lattice_bounds()

        if len(p.shape) != 1 or p.shape[0] != self.parent.dimension:
            raise ValueError(f"Invalid input shape, p should be {self.parent.dimension}-dimensional")

        if p.dtype != 'int':
            raise ValueError(f"p should be an int vector!")

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

    def _find_correspondence(self, other: "LatticeTensor") -> Optional[Dict]:
        """
        This is mentioned briefly in the definition about compatibility, but
        we make it concrete here. If you provide the method an invalid
        combination of lattices, it raises an error (you shouldn't compare
        two lattice tensors on different lattices this /should/ be
        programmer error).

        Otherwise it returns a correspondence dictionary

        :param other: Another lattice tensor on the same lattice
        :return: A dict with d[self.coset_index] = other.coset_index or
        """
        if len(self._coset_offsets) != len(other._coset_offsets):
            raise ValueError("LatticeTensors must belong to the same lattice")

        if len(self._coset_offsets) == 1:
            return [0]

        c_a = sum(self._coset_offsets) / len(self._coset_offsets)
        c_b = sum(other._coset_offsets) / len(other._coset_offsets)

        a = [_ - c_a for _ in self._coset_offsets]
        b = [_ - c_b for _ in other._coset_offsets]

        corresp = [0] * len(a)
        found = 0
        for i, p in enumerate(a):
            for j, q in enumerate(b):
                if np.abs(p - q).sum() < self.eps:
                    if tuple(self._cosets[i].shape) == tuple(other._cosets[j].shape):
                        corresp[i] = j
                        found += 1
                        break
        if found != len(a):
            return None
        return corresp

    def __cmp__(self, other):
        raise NotImplementedError()

    def __add__(self, other):
        """
        Adds two lattice tensors, wraps the result in a new lattice tensor.

        :param other: Another lattice tensor on the same lattice
        :return: A lattice tensor
        """
        correspondence = self._find_correspondence(other)
        keys = {}
        for i, (offset, coset_a) in enumerate(zip(self._coset_offsets, self._cosets)):
            offset = tuple([int(_) for _ in offset])
            keys[tuple(offset)] = coset_a + other._cosets[correspondence[i]]
        return self.parent(keys)

    def __sub__(self, other):
        """
        Subtracts two lattice tensors, wraps the result in a new lattice tensor.

        :param other: Another lattice tensor on the same lattice
        :return: A lattice tensor
        """
        correspondence = self._find_correspondence(other)
        keys = {}
        for i, (offset, coset_a) in enumerate(zip(self._coset_offsets, self._cosets)):
            offset = tuple([int(_) for _ in offset])
            keys[tuple(offset)] = coset_a - other._cosets[correspondence[i]]
        return self.parent(keys)

    def __mul__(self, other):
        """
        Multiplies two lattice tensors, wraps the result in a new lattice tensor.
        
        :param other: Another lattice tensor on the same lattice
        :return: A lattice tensor
        """
        correspondence = self._find_correspondence(other)
        keys = {}
        for i, (offset, coset_a) in enumerate(zip(self._coset_offsets, self._cosets)):
            offset = tuple([int(_) for _ in offset])
            keys[tuple(offset)] = coset_a * other._cosets[correspondence[i]]
        return self.parent(keys)

    def raw_coset_point_info(self, lattice_site):
        coset = self.parent.coset_index(lattice_site)
        if coset is None:
            return None
        return coset, (lattice_site - self.coset_vectors[coset])//self.parent.coset_scale

    @staticmethod
    def _project_axis_bounds(coset_vector, coset_scale, element_count, axis):
        start = coset_vector[axis]
        end = coset_scale[axis] * element_count + coset_vector[axis]
        return start, end


class Lattice:
    """
    The general "LatticeTensor" factory. Basically, this holds all the
    important information about the point structure of any Lattice.
    LatticeTensors are instances of multi-dimensional sequences, but they
    store values only within a bounded region (and on an integer lattice).

    An instance of this class /creates/ lattice tensors from a collection
    of tensors (and possibly shifts, but if no shifts are spllied, then
    it is assumed that the default coset structure (all positive) is used)
    """

    def __init__(self,
                 input_lattice: Union[List, str],
                 scale: Union[np.array, None] = None,
                 tensor_backend: torch._tensor = torch.Tensor):
        """
        Instantiate the LatticeTensor factory.

        :param input_lattice: Either a lattice name (i.e. quincunx, qc, bcc
                    etc..) or a list of coset vectors.

        :param scale: If input lattice is a list of coset vectors, then this
                    should be an n-dimensional integer vector specifying the
                    scale of each coset.

        :param tensor_backend: The type of tensor that backs this tensor
                    created from this factory. This will mostly be torch.Tensor,
                    but can be any torch._tensor type.
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
        if any([not isinstance(_, (np.ndarray, np.generic))  for _ in coset]):
            raise ValueError(f"The input 'coset' should be a non-empty list of np.array")

        if any([_.size != coset[0].size for _ in coset]):
            raise ValueError(f"The input 'coset' should be a non-empty list of 1-D IntTensors with the same size")

        self.tensor = tensor_backend
        self._dimension = coset[0].shape[0]
        self._coset_scale = scale

        canonical_offsets = [_ % np.abs(scale) for _ in coset]

        self._coset_vectors = sorted(canonical_offsets,
                                     key=lambda x: sum([abs(_.item()) for _ in x[:]])
            )

    def _validate_coset_vectors(self, vectors):
        pass

    @property
    def coset_vectors(self):
        """
        The coset vectors $v_i mod D$ of the current lattice.
        """
        return self._coset_vectors[:]

    @property
    def dimension(self) -> int:
        """
        The dimension of the lattice.
        """
        return self._dimension

    @property
    def coset_scale(self) -> np.array:
        """
        The coset scale $D$ of the lattice
        """
        return self._coset_scale

    @property
    def coset_count(self) -> int:
        """
        The total number of cosets for the lattice
        """
        return len(self._coset_vectors)

    def coset_index(self, pt: Union[Tuple[int], List[int], np.array]) -> Union[int, None]:
        return self.cartesian_index(pt)[0]

    def cartesian_index(self, pt: Union[Tuple[int], List[int], np.array]):
        if type(pt) == tuple or type(pt) == list:
            pt = np.array([int(_) for _ in pt], dtype='int')
        for idx, c in enumerate(self._coset_vectors):
            x = (1 / self._coset_scale) * (pt - c)
            if all([np.abs(_.round() - _) < 2 * torch.finfo().eps for _ in x]):
                return idx, x
        return None, None

    def coset_offset(self, idx):
        if 0 > idx <= self.coset_count:
            raise ValueError('Invalid coset offset index!')
        return self._coset_vectors[idx].astype('int')

    def cartesian_to_lattice(self, pt, coset_index):
        return (self._coset_scale * np.array(pt, dtype='int') + self._coset_vectors[coset_index]).astype('int')

    def __cmp__(self, other):
        raise NotImplementedError()

    def __eq__(self, other):
        eq = all([np.abs(a - b).sum() < 1e-5 for a,b in zip(self.coset_vectors, other.coset_vectors)])
        return eq and all([np.abs(a - b).sum() < 1e-5 for a,b in zip(self.coset_scale, other.coset_scale)])

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

            vectors = [np.array(_, dtype='int') for _ in args[0]]
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
