
try:                        
    # ➊ Prefer JAX if it is available
    import jax.numpy as xp
    import jax
    
    JAX = True
    
    # Imports
    from jax import tree_util, lax
    from flax.struct import dataclass, field            # immutable, PyTree-aware
    
    dataclass = dataclass
    field = field
    tree_leaves = tree_util.tree_leaves
    tree_map = tree_util.tree_map
    dynamic_update_slice = lax.dynamic_update_slice
    dynamic_slice = lax.dynamic_slice
    full = jax.numpy.full
    arange = jax.numpy.arange
    squeeze = jax.numpy.squeeze
    # Random number generation
    
    class random:
        @staticmethod
        def normal(key, shape):
            return jax.random.normal(key, shape)
        @staticmethod
        def split(key, n):
            return jax.random.split(key, n)
        @staticmethod
        def fold_in(key, i):
            """Fold in an integer to the random key."""
            return jax.random.fold_in(key, i)
        @staticmethod
        def PRNGKey(seed):
            """Create a new random key."""
            return jax.random.PRNGKey(seed)
        
      
except ModuleNotFoundError: # ➋ Pure-NumPy fallback
    import numpy as xp
    import policy_package.observation.numpy_jax_shim as numpy_jax_shim
    
    dataclass = numpy_jax_shim.dataclass
    field = numpy_jax_shim.field
    tree_leaves = numpy_jax_shim.np_tree_leaves
    tree_map = numpy_jax_shim.np_tree_map
    dynamic_update_slice = numpy_jax_shim.dynamic_update_slice
    dynamic_slice = numpy_jax_shim.dynamic_slice
    full = xp.full
    arange = xp.arange
    squeeze = xp.squeeze
    
    random = numpy_jax_shim.random

    JAX = False
    