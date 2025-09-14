from typing import Tuple
import numpy as np

def is_within_bounds(value: np.ndarray, bounds: Tuple[np.ndarray, np.ndarray]) -> np.bool_:
        return np.all(value >= bounds[0]) and np.all(value <= bounds[1])