from abc import ABC
from enum import Enum
import math
from typing import TYPE_CHECKING
from typing_extensions import Annotated, Self
import bounce_policy_pkg.observation.observation_backend as bc

if TYPE_CHECKING:
    from environments.drone_env_refactor_hover_only.types.augmented_pipeline_state import AugmentedPipelineState

class StrEnum(str, Enum):
    pass

Vec3  = Annotated[bc.xp.ndarray, (3,)]
Vec4  = Annotated[bc.xp.ndarray, (4,)]
Quat  = Annotated[bc.xp.ndarray, (4,)]
Flag1 = Annotated[bc.xp.ndarray, (1,), bool]

# --- Instant Observation ---
class ObservationType(StrEnum):
    DRONE_VICON = "drone_vicon"
    DRONE_BALL_RELATIVE_VICON = "drone_ball_vicon"
    
@bc.dataclass
class Observation(ABC):
    def validate(self) -> bool:
        return self._validate(self)

    @classmethod
    def _validate(cls, obs) -> bool:
        """Validate the observation follows the type hints (shape and dtype)."""
        assert isinstance(obs, cls), f"Observation must be an instance of {cls.__name__}, got {type(obs).__name__} instead."
        for name, anno in cls.__annotations__.items():
            field_type = anno.__origin__
            field_shape = anno.__metadata__[0]

            if not hasattr(obs, name):
                raise ValueError(f"Observation is missing field: {name}")

            if not isinstance(getattr(obs, name), field_type):
                raise TypeError(f"Field '{name}' must be of type {field_type}, "
                                f"got {type(getattr(obs, name))} instead.")

            if getattr(obs, name).shape != field_shape:
                raise ValueError(f"Field '{name}' must have shape {field_shape}, "
                                 f"got {getattr(obs, name).shape} instead.")
        return True
    
    @classmethod
    def generate_zero(cls) -> Self:
        """Generate an observation with all fields set to zero."""
        return cls(**{field_name: bc.xp.zeros(anno.__metadata__[0], dtype=anno.__metadata__[1] if 1 < len(anno.__metadata__) else bc.xp.float32) for field_name, anno in cls.__annotations__.items()})

    @classmethod
    def generate_random(cls, key: bc.xp.ndarray) -> Self:
        random_attr = {}
        for (field_name, anno), k in zip(cls.__annotations__.items(), bc.random.split(key, len(cls.__annotations__))):
            shape = anno.__metadata__[0]
            random_attr[field_name] = bc.random.normal(k, shape)

        return cls(**random_attr)
    
    def to_array(self) -> bc.xp.ndarray:
        """Serialize the observation to a 1D array."""
        arrays = []
        for field_name in self.__annotations__:
            field_data = getattr(self, field_name).ravel()
            if field_data.dtype != bc.xp.float32:
                field_data = field_data.astype(bc.xp.float32)
            arrays.append(field_data)
        return bc.xp.concatenate(arrays)

    @classmethod
    def from_array(cls, array: bc.xp.ndarray) -> Self:
        """Deserialize a 1D array back into the observation."""
        out = {}
        idx = 0
        for field_name, anno in cls.__annotations__.items():
            field_shape = anno.__metadata__[0]
            field_size = math.prod(field_shape)
            field_data = array[idx:idx + field_size].reshape(field_shape)
            out[field_name] = field_data
            idx += field_size
        return cls(**out)
    
    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Observation):
            return NotImplemented
        return all(bc.xp.array_equal(getattr(self, field.name), getattr(other, field.name)) for field in bc.fields(self))

    @classmethod
    def from_state(cls, state: 'AugmentedPipelineState') -> Self:
        """
        Create an observation from the current state.
        This method should be implemented by subclasses.
        """
        raise NotImplementedError
    
@bc.dataclass(eq=False)
class DroneViconObs(Observation):
    """Current observation of the drone environment with Vicon data."""
    drone_position: Vec3
    drone_orientation: Quat
    drone_velocity: Vec3
    drone_body_rate: Vec3
    previous_action: Vec4

@bc.dataclass(eq=False)
class DroneBallRelativeViconObs(Observation):
    """Current observation of the drone environment."""
    # --- Drone ---
    drone_position: Vec3
    drone_orientation: Quat
    drone_velocity: Vec3
    drone_body_rate: Vec3
    previous_action: Vec4
    
    # --- Ball ---
    ball_velocity: Vec3
    ball_relative_to_drone: Vec3
    
    
if __name__ == "__main__":
    
    obs = FullObservationWithViconFlag(
        drone_position=bc.xp.zeros((3,)),
        drone_orientation=bc.xp.zeros((4,)),
        drone_velocity=bc.xp.zeros((3,)),
        drone_body_rate=bc.xp.zeros((3,)),
        previous_action=bc.xp.zeros((4,))
    )
    
    
    zero = FullObservationWithViconFlag.generate_zero()