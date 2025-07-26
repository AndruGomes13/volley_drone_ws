from abc import ABC, abstractmethod
from dataclasses import dataclass, fields
import functools
import math
from typing import ClassVar, Optional, Tuple, Union
from typing_extensions import Annotated, Self, get_origin, get_args

from state_interface import StateHistory, StateHistoryNumpy

Shape = Tuple[int, ...]
import numpy as np

POS_CLIP_BOUNDS = (np.array((-7, -2.5, 0)), np.array((7, 2.5, 100)))
RELATIVE_POS_CLIP_BOUNDS = (np.array((-10.0, -10.0, -10.0)), np.array((10.0, 10.0, 10.0)))

def observation_dataclass(cls):
    """Decorator that makes the obs dataclasses and builds the __FIELDS__"""
    derived: list[tuple[str, Shape]] = []
    for f in fields(cls):
        anno = cls.__annotations__[f.name]
        if get_origin(anno) is Annotated:
            base, shape = get_args(anno)
            if base is np.ndarray:
                derived.append((f.name, shape))

    cls.__FIELD_SPECS__ = tuple(derived)
    return cls

@dataclass(frozen=True)
class Observation(ABC):
    __FIELD_SPECS__: ClassVar[Tuple[Tuple[str, Tuple[Union[int, str], ...]]]] = ()
    __FIELDS__: ClassVar[Tuple[Tuple[str, Shape]]] = tuple()
    ACTION_HISTORY_LEN: ClassVar[int] = 1
    DRONE_HISTORY_LEN: ClassVar[int] = 1
    BALL_HISTORY_LEN: ClassVar[int] = 1

    __resolved__: ClassVar[bool] = False

    @classmethod
    @abstractmethod
    def get_observation(cls, state_history: StateHistory) -> Self:
        pass

    def to_array(self) -> np.ndarray:
        """Flattens the observation into a 1D array using the static field specification."""
        if not self.__class__.__resolved__:
            raise RuntimeError("You must call .resolve_fields(...) before using to_array()")

        arrays = []
        for field_name, _ in self.__FIELDS__:
            arrays.append(getattr(self, field_name).ravel())
        return np.concatenate(arrays, axis=0)

    @classmethod
    def from_array(cls, arr: np.ndarray):
        """Rebuilds an observation from the 1D array using the static field specification."""
        if not cls.__resolved__:
            raise RuntimeError("You must call .resolve_fields(...) before using from_array()")

        out = {}
        idx = 0
        for field_name, shape in cls.__FIELDS__:
            size = math.prod(shape)  # Compute the total size for this field.
            out[field_name] = arr[idx : idx + size].reshape(shape)
            idx += size

        assert idx == arr.size, f"got {arr.size} but expected {idx}"
        return cls(**out)

    @classmethod
    def resolve_fields(cls, drone_state_history_len: Optional[int] = None, ball_state_history_len: Optional[int] = None, action_history_len:Optional[int] = None):
        if drone_state_history_len is not None:
            cls.DRONE_HISTORY_LEN = drone_state_history_len
        if ball_state_history_len is not None:
            cls.BALL_HISTORY_LEN = ball_state_history_len
        if action_history_len is not None:
            cls.ACTION_HISTORY_LEN = action_history_len

        resolved = []
        for name, spec in cls.__FIELD_SPECS__:
            # turn any string dims into ints
            shape = tuple(getattr(cls, dim) if isinstance(dim, str) else dim for dim in spec)
            resolved.append((name, shape))
        cls.__FIELDS__ = tuple(resolved)
        cls.__resolved__ = True

        # wrap get_observation to auto‐validate
        orig = cls.get_observation.__func__  # unwrap the classmethod

        @functools.wraps(orig)
        def _wrapped(cls, state):
            obs = orig(cls, state)
            cls._validate(obs)
            return obs

        cls.get_observation = classmethod(_wrapped)

    @classmethod
    def _validate(cls, obs: Self):
        if not cls.__resolved__:
            raise RuntimeError("Must call resolve_fields(...) before using get_observation")
        for name, expected in cls.__FIELDS__:
            actual = getattr(obs, name).shape
            assert actual == expected, f"{cls.__name__}.{name} shape mismatch: got {actual}, expected {expected}"

    @classmethod
    def generate_random(cls, key: np.ndarray) -> Self:
        if not cls.__resolved__:
            raise RuntimeError("Must call resolve_fields(...) before using get_observation")
        
        random_attr = {}
        for (field_name, shape) in cls.__FIELDS__:
            random_attr[field_name] = np.random.normal(size=shape)

        # print(random_attr.keys())
        return cls(**random_attr)

    @classmethod
    def get_observation_shape(cls) -> Tuple[int, ...]:
        """Returns the shape of the observation as a 1D array."""
        if not cls.__resolved__:
            raise RuntimeError("You must call .resolve_fields(...) before using get_observation_shape()")

        flattened_sizes = [math.prod(shape) for _, shape in cls.__FIELDS__]
        shape = (sum(flattened_sizes), )
        return shape

    def __eq__(self, other: Self) -> bool:
        for field_name, shape in self.__FIELDS__:
            self_value: np.ndarray = getattr(self, field_name)
            other_value: np.ndarray = getattr(other, field_name)

            # Check shape
            if self_value.shape != other_value.shape:
                return False

            # Check type
            if self_value.dtype != other_value.dtype:
                return False

            # Check if equal
            if not np.allclose(self_value, other_value):
                return False

        return True

@observation_dataclass
@dataclass(frozen=True, eq=False)
class FullObservationWithBall(Observation):
    # --- Drone ---
    drone_position: Annotated[np.ndarray, ("DRONE_HISTORY_LEN",3)]
    drone_orientation_wxyz: Annotated[np.ndarray, ("DRONE_HISTORY_LEN",4)]
    drone_velocity: Annotated[np.ndarray, ("DRONE_HISTORY_LEN",3)]
    drone_body_rate: Annotated[np.ndarray, ("DRONE_HISTORY_LEN",3)]

    # --- Ball ---
    ball_position: Annotated[np.ndarray, ("BALL_HISTORY_LEN",3)]
    ball_velocity: Annotated[np.ndarray, ("BALL_HISTORY_LEN",3)]
    ball_relative_position: Annotated[np.ndarray, (3,)]

    # --- Last action ---
    last_policy_action: Annotated[np.ndarray, ("ACTION_HISTORY_LEN",4)] #NOTE: With the current SBUS implementation, no motor speed is available.

    @classmethod
    def get_observation(cls, state_history: StateHistory) -> Self:
        state_history_numpy = StateHistoryNumpy.from_state_history(state_history, drone_history_length=cls.DRONE_HISTORY_LEN, ball_history_length=cls.BALL_HISTORY_LEN, action_history_length=cls.ACTION_HISTORY_LEN)

        ball_pos_clipped = np.clip(state_history_numpy.ball_state_history.position, POS_CLIP_BOUNDS[0], POS_CLIP_BOUNDS[1])
        
        ball_pos_relative = state_history_numpy.ball_state_history.position[0] - state_history_numpy.ball_state_history.position[0]

        # Clip relative ball pos
        ball_pos_relative_clipped = np.clip(ball_pos_relative, RELATIVE_POS_CLIP_BOUNDS[0], RELATIVE_POS_CLIP_BOUNDS[1])

        return cls(
            drone_position=state_history_numpy.drone_state_history.position,
            drone_orientation_wxyz=state_history_numpy.drone_state_history.orientation_wxyz,
            drone_velocity=state_history_numpy.drone_state_history.velocity,
            drone_body_rate=state_history_numpy.drone_state_history.body_rate,
            ball_relative_position=ball_pos_relative_clipped,
            ball_position=ball_pos_clipped,
            ball_velocity=state_history_numpy.ball_state_history.velocity,
            last_policy_action=state_history_numpy.action_history,
        )


if __name__ == "__main__":
    FullObservationWithBall.resolve_fields(20,10, 5)
    # print(FullObservationWithBall.__FIELDS__)
        


    
    
    