#!/usr/bin/env python3
# Copyright 2025 Litianyu141
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
PyTree utilities for PyTorch, closely following JAX's implementation.
This module provides tree manipulation functions that are compatible with torch.jit.

See the `JAX pytrees note <pytrees.html>`_
for examples.
"""

import torch
import collections
from typing import Any, Callable, List, Optional, Tuple, Union, NamedTuple
from functools import partial
import operator


class PyTreeDef:
    """Represents the structure of a PyTree."""
    
    def __init__(self, structure: Any, num_leaves: int = 0, num_nodes: int = 0):
        self.structure = structure
        self.num_leaves = num_leaves
        self.num_nodes = num_nodes
    
    def __repr__(self):
        return f"PyTreeDef({self.structure})"
    
    def children(self) -> List['PyTreeDef']:
        """Return a list of PyTreeDefs for immediate children."""
        children = []
        if isinstance(self.structure, collections.OrderedDict):
            for v in self.structure.values():
                children.append(PyTreeDef(v))
        elif isinstance(self.structure, (tuple, list)):
            if isinstance(self.structure, tuple) and len(self.structure) == 2 and isinstance(self.structure[0], type):
                # NamedTuple case
                _, field_structures = self.structure
                for fs in field_structures:
                    children.append(PyTreeDef(fs))
            else:
                for v in self.structure:
                    children.append(PyTreeDef(v))
        return children
    
    def unflatten(self, leaves: List[Any]) -> Any:
        """Reconstruct a PyTree from leaves using this structure."""
        return tree_unflatten(self, leaves)


def _is_leaf(x: Any) -> bool:
    """Check if an object is a leaf (not a container)."""
    return not isinstance(x, (list, tuple, dict))


def tree_flatten(tree: Any) -> Tuple[List[Any], PyTreeDef]:
    """
    Flattens a PyTree into a list of leaves and a structure definition.
    
    Args:
        tree: The PyTree to flatten
        
    Returns:
        A tuple of (leaves, treedef)
    """
    leaves = []
    
    def _flatten_helper(subtree):
        if isinstance(subtree, dict):
            result = collections.OrderedDict()
            for k in sorted(subtree.keys()):
                v = subtree[k]
                if _is_leaf(v):
                    leaves.append(v)
                    result[k] = None  # Leaf placeholder
                else:
                    result[k] = _flatten_helper(v)
            return result
        elif isinstance(subtree, tuple):
            if hasattr(subtree, '_fields'):  # NamedTuple
                result = (type(subtree), tuple(_flatten_helper(v) for v in subtree))
            else:
                result = tuple(_flatten_helper(v) for v in subtree)
            return result
        elif isinstance(subtree, list):
            return [_flatten_helper(v) for v in subtree]
        else:
            # Leaf node
            leaves.append(subtree)
            return None
    
    structure = _flatten_helper(tree)
    num_leaves = len(leaves)
    num_nodes = _count_nodes(structure)
    return leaves, PyTreeDef(structure, num_leaves, num_nodes)


def _count_nodes(structure: Any) -> int:
    """Count the number of nodes in a tree structure."""
    if structure is None:
        return 1  # Leaf node
    elif isinstance(structure, collections.OrderedDict):
        return 1 + sum(_count_nodes(v) for v in structure.values())
    elif isinstance(structure, tuple):
        if len(structure) == 2 and isinstance(structure[0], type):
            # NamedTuple case
            _, field_structures = structure
            return 1 + sum(_count_nodes(fs) for fs in field_structures)
        else:
            return 1 + sum(_count_nodes(v) for v in structure)
    elif isinstance(structure, list):
        return 1 + sum(_count_nodes(v) for v in structure)
    else:
        return 1


def tree_unflatten(treedef: PyTreeDef, leaves: List[Any]) -> Any:
    """
    Reconstructs a PyTree from leaves and structure definition.
    
    Args:
        treedef: The structure definition
        leaves: List of leaf values
        
    Returns:
        The reconstructed PyTree
    """
    leaf_iter = iter(leaves)
    
    def _unflatten_helper(structure):
        if isinstance(structure, collections.OrderedDict):
            result = collections.OrderedDict()
            for k, v in structure.items():
                if v is None:  # Leaf placeholder
                    try:
                        result[k] = next(leaf_iter)
                    except StopIteration:
                        raise ValueError("Not enough leaves for tree structure")
                else:
                    result[k] = _unflatten_helper(v)
            return result
        elif isinstance(structure, tuple):
            if len(structure) == 2 and isinstance(structure[0], type) and hasattr(structure[0], '_fields'):
                # NamedTuple reconstruction
                cls, field_structures = structure
                fields = tuple(_unflatten_helper(fs) for fs in field_structures)
                return cls(*fields)
            else:
                return tuple(_unflatten_helper(v) for v in structure)
        elif isinstance(structure, list):
            return [_unflatten_helper(v) for v in structure]
        elif structure is None:
            # Leaf placeholder
            try:
                return next(leaf_iter)
            except StopIteration:
                raise ValueError("Not enough leaves for tree structure")
        else:
            raise ValueError(f"Unexpected structure type: {type(structure)}")
    
    result = _unflatten_helper(treedef.structure)
    
    # Check that all leaves were consumed
    try:
        next(leaf_iter)
        raise ValueError("Too many leaves for tree structure")
    except StopIteration:
        pass
    
    return result


def tree_leaves(tree: Any) -> List[Any]:
    """Extract all leaves from a PyTree."""
    leaves, _ = tree_flatten(tree)
    return leaves


def tree_structure(tree: Any) -> PyTreeDef:
    """Get the structure definition of a PyTree."""
    _, treedef = tree_flatten(tree)
    return treedef


def tree_map(func: Callable, tree: Any, *rest: Any) -> Any:
    """
    Map a function over the leaves of one or more PyTrees.
    
    Args:
        func: Function to apply to leaves
        tree: Primary PyTree
        *rest: Additional PyTrees with compatible structure
        
    Returns:
        New PyTree with same structure as input trees
    """
    leaves, treedef = tree_flatten(tree)
    
    # Flatten all other trees and check compatibility
    other_leaves_lists = []
    for other_tree in rest:
        other_leaves, other_treedef = tree_flatten(other_tree)
        if len(other_leaves) != len(leaves):
            raise ValueError(f"tree_map requires trees with same number of leaves")
        other_leaves_lists.append(other_leaves)
    
    # Apply function to corresponding leaves
    new_leaves = []
    for i, leaf in enumerate(leaves):
        args = [leaf] + [other_leaves[i] for other_leaves in other_leaves_lists]
        new_leaves.append(func(*args))
    
    return tree_unflatten(treedef, new_leaves)


def tree_reduce(func: Callable, tree: Any, initializer: Any = None) -> Any:
    """
    Reduce over the leaves of a PyTree.
    
    Args:
        func: Binary function for reduction
        tree: PyTree to reduce over
        initializer: Optional initial value
        
    Returns:
        Reduced value
    """
    leaves = tree_leaves(tree)
    
    if not leaves:
        if initializer is None:
            raise ValueError("Cannot reduce empty tree without initializer")
        return initializer
    
    if initializer is None:
        result = leaves[0]
        start_idx = 1
    else:
        result = initializer
        start_idx = 0
    
    for leaf in leaves[start_idx:]:
        result = func(result, leaf)
    
    return result


def tree_all(tree: Any) -> bool:
    """Check if all leaves in PyTree are truthy."""
    leaves = tree_leaves(tree)
    return all(bool(leaf.all().item()) if isinstance(leaf, torch.Tensor) else bool(leaf) for leaf in leaves)


# Partial class for JAX compatibility
class Partial:
    """A partial function that mimics JAX's Partial."""
    def __init__(self, func, *args, **kwargs):
        self.func = func
        self.args = args
        self.kwargs = kwargs
    
    def __call__(self, *more_args, **more_kwargs):
        combined_kwargs = {**self.kwargs, **more_kwargs}
        return self.func(*(self.args + more_args), **combined_kwargs)


def tree_sub(tree1: Any, tree2: Any) -> Any:
    """Element-wise subtraction of two PyTrees."""
    return tree_map(torch.sub, tree1, tree2)


def tree_mul(scalar: Union[float, int, torch.Tensor], tree: Any) -> Any:
    """Scalar multiplication of a PyTree."""
    if isinstance(scalar, torch.Tensor) and scalar.numel() != 1:
        raise ValueError("Scalar must be a single value")
    
    return tree_map(lambda x: scalar * x, tree)


def tree_zeros_like(tree: Any) -> Any:
    """Create a PyTree of zeros with the same structure and dtypes."""
    return tree_map(torch.zeros_like, tree)


def tree_ones_like(tree: Any) -> Any:
    """Create a PyTree of ones with the same structure and dtypes."""
    return tree_map(torch.ones_like, tree)


def tree_conj(tree: Any) -> Any:
    """Complex conjugate of a PyTree."""
    return tree_map(torch.conj, tree)


def tree_real(tree: Any) -> Any:
    """Real part of a PyTree."""
    return tree_map(lambda x: x.real if torch.is_complex(x) else x, tree)


def tree_imag(tree: Any) -> Any:
    """Imaginary part of a PyTree."""
    return tree_map(lambda x: x.imag if torch.is_complex(x) else torch.zeros_like(x), tree)


@torch.jit.script
def _tensor_vdot_real_part(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Compute real part of vdot for complex tensors, handling both real and complex cases."""
    if torch.is_complex(x) or torch.is_complex(y):
        # For complex tensors, compute Re(x^H * y) = Re(x) * Re(y) + Im(x) * Im(y)
        x_real = x.real if torch.is_complex(x) else x
        x_imag = x.imag if torch.is_complex(x) else torch.zeros_like(x)
        y_real = y.real if torch.is_complex(y) else y
        y_imag = y.imag if torch.is_complex(y) else torch.zeros_like(y)
        return torch.sum(x_real * y_real + x_imag * y_imag)
    else:
        return torch.sum(x * y)


def tree_vdot_real(tree1: Any, tree2: Any) -> torch.Tensor:
    """
    Compute real part of vector dot product for PyTrees.
    This follows JAX's _vdot_real_tree implementation.
    """
    leaves1 = tree_leaves(tree1)
    leaves2 = tree_leaves(tree2)
    
    if len(leaves1) != len(leaves2):
        raise ValueError("Trees must have same structure for vdot")
    
    if not leaves1:
        return torch.tensor(0.0, dtype=torch.float64)
    
    # Get device and appropriate dtype from first leaves
    device = leaves1[0].device
    # Use highest precision available - torch.float64
    dtype = torch.float64
    
    total = torch.tensor(0.0, dtype=dtype, device=device)
    for x, y in zip(leaves1, leaves2):
        x_flat = x.reshape(-1)
        y_flat = y.reshape(-1)
        total += _tensor_vdot_real_part(x_flat, y_flat)
    
    return total


def tree_vdot(tree1: Any, tree2: Any) -> torch.Tensor:
    """Compute vector dot product for PyTrees."""
    leaves1 = tree_leaves(tree1)
    leaves2 = tree_leaves(tree2)
    
    if len(leaves1) != len(leaves2):
        raise ValueError("Trees must have same structure for vdot")
    
    if not leaves1:
        return torch.tensor(0.0, dtype=torch.complex128)
    
    # Get device and dtype from first leaves
    device = leaves1[0].device
    # Determine result dtype (complex if any input is complex)
    is_complex = any(torch.is_complex(x) for x in leaves1 + leaves2)
    if is_complex:
        dtype = torch.complex128  # Highest precision complex
    else:
        dtype = torch.float64     # Highest precision real
    
    total = torch.tensor(0.0, dtype=dtype, device=device)
    for x, y in zip(leaves1, leaves2):
        x_flat = x.reshape(-1)
        y_flat = y.reshape(-1)
        total += torch.vdot(x_flat, y_flat)
    
    return total


def tree_norm(tree: Any) -> torch.Tensor:
    """Compute L2 norm of a PyTree."""
    norm_sq = tree_vdot_real(tree, tree)
    return torch.sqrt(norm_sq)


def tree_dot(tree1: Any, tree2: Any) -> Any:
    """Element-wise dot product for PyTrees (tensor contraction on last axis)."""
    return tree_map(lambda x, y: torch.dot(x.reshape(-1), y.reshape(-1)), tree1, tree2)


def tree_all(tree: Any) -> bool:
    """Check if all leaves in PyTree are truthy."""
    leaves = tree_leaves(tree)
    return all(bool(leaf.all().item()) if isinstance(leaf, torch.Tensor) else bool(leaf) for leaf in leaves)
