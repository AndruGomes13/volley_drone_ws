import dataclasses
from dataclasses import field as _field
import numpy as _np
from typing import Any, Callable, Tuple

def dataclass(_cls=None, **kwargs):
    """ Decorator to convert the default dataclass implementation to be more similar to flax's dataclass."""
    def wrap(cls):
        if 'frozen' not in kwargs:
            kwargs['frozen'] = True
        
        cls = dataclasses.dataclass(cls, **kwargs)
        
        def replace(self, **changes):
            return dataclasses.replace(self, **changes)

        cls.replace = replace
        return cls

    if _cls is None:
        return wrap
    return wrap(_cls)

def field(**kwargs):
    ''' Make this function accept the pytree_node argument, but ignore it since we don't need it for the NumPy backend. '''
    del kwargs['pytree_node']  # Ignore pytree_node argument
    return _field(**kwargs)

def np_is_leaf(x: Any) -> bool:
    return isinstance(x, _np.ndarray)

def np_tree_leaves(tree):
    if np_is_leaf(tree):
        return [tree]
    if _dc.is_dataclass(tree):
        leaves = []
        for f in _dc.fields(tree):
            leaves += np_tree_leaves(getattr(tree, f.name))
        return leaves
    if isinstance(tree, (list, tuple)):
        return [l for t in tree for l in np_tree_leaves(t)]
    if isinstance(tree, dict):
        return [l for v in tree.values() for l in np_tree_leaves(v)]
    raise TypeError(f"Unsupported container: {type(tree).__name__}")

def np_tree_map(fn: Callable[..., Any], tree, *rest):
    if np_is_leaf(tree):
        return fn(tree, *(r for r in rest))
    if dataclasses.is_dataclass(tree):
        kwargs = {f.name: np_tree_map(fn,
                                getattr(tree, f.name),
                                *(getattr(r, f.name) for r in rest))
                for f in dataclasses.fields(tree)}
        return type(tree)(**kwargs)
    if isinstance(tree, (list, tuple)):
        return type(tree)(np_tree_map(fn, t, *(r[i] for r in rest))
                        for i, t in enumerate(tree))
    if isinstance(tree, dict):
        return {k: np_tree_map(fn, v, *(r[k] for r in rest))
                for k, v in tree.items()}
    raise TypeError(f"Unsupported container: {type(tree).__name__}")

def _array_dynamic_update_slice(a: _np.ndarray,
                                b: _np.ndarray,
                                start: Tuple[int, ...]) -> _np.ndarray:
    """Immutable NumPy equivalent of lax.dynamic_update_slice on one array."""
    out = a.copy()
    slices = tuple(slice(s, s + d) for s, d in zip(start, b.shape))
    out[slices] = b
    return out

def _array_dynamic_slice(a: _np.ndarray,
                         start: Tuple[int, ...],
                         size:  Tuple[int, ...]) -> _np.ndarray:
    """Immutable NumPy equivalent of lax.dynamic_slice on one array."""
    slices = tuple(slice(s, s + d) for s, d in zip(start, size))
    return a[slices]
    
def dynamic_update_slice(a, b, start):
    """
    JAX-style `dynamic_update_slice`, but works for
      • a single ndarray, **or**
      • any tree whose leaves are ndarrays.
    """
    if np_is_leaf(a):                                  # plain ndarray
        return _array_dynamic_update_slice(a, b, start)
    
    # tree case -------------------------------------------------------------
    def _update(buf_leaf, val_leaf):
        leaf_start = start + (0,) * (buf_leaf.ndim - len(start))
        return _array_dynamic_update_slice(buf_leaf, val_leaf, leaf_start)
    
    return np_tree_map(_update, a, b)

def dynamic_slice(a, start, size):
    """
    JAX-style `dynamic_slice`, supporting both ndarrays and trees of ndarrays.
    """
    if np_is_leaf(a):                                  # plain ndarray
        return _array_dynamic_slice(a, start, size)
    
    # tree case -------------------------------------------------------------
    def _slice(buf_leaf):
        leaf_start = start + (0,) * (buf_leaf.ndim - len(start))
        leaf_size  = size  + buf_leaf.shape[len(size):]    # keep full trailing dims
        return _array_dynamic_slice(buf_leaf, leaf_start, leaf_size)
    
    return np_tree_map(_slice, a)


class random:
    @staticmethod
    def normal(key, shape, dtype=_np.float32):
        _np.random.seed(key)
        return _np.random.normal(size=shape).astype(dtype)

    @staticmethod
    def split(key, n):
        _np.random.seed(key)  # Set seed for reproducibility
        return [_np.random.randint(0, 2**32, size=(1,)).astype(_np.uint32) for _ in range(n)]

    @staticmethod
    def fold_in(key, i):
        """Fold in an integer to the random key."""
        _np.random.seed(key + i)
        return _np.random.randint(0, 2**32, size=(1,)).astype(_np.uint32)
    
    @staticmethod
    def PRNGKey(seed):
        """Create a new random key."""
        return seed if isinstance(seed, _np.ndarray) else _np.array(seed, dtype=_np.int32)