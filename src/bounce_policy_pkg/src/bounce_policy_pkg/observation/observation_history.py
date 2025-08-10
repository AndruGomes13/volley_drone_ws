from dataclasses import fields
import math
from typing import TYPE_CHECKING, Dict, Generic, Protocol, Type, TypeVar, get_type_hints
from typing_extensions import Self
from bounce_policy_pkg.types.buffer import HistBuf
import bounce_policy_pkg.observation.observation_backend as bc

if TYPE_CHECKING:
    from observation import Observation

T = TypeVar('T', bound='Observation')

@bc.dataclass
class ObservationHistory(Generic[T], Protocol):
    """
    Protocol for observation history classes that can handle
    a history of observations with specified lengths.
    """
    ObservationBase: Type[T]

    @classmethod
    def init(cls, example: T) -> Self:
        """Initialize the history buffers with specified lengths."""
        ...

    def push(self, inst: T) -> Self:
        """Push an instance of the base class into the history buffers."""
        ...

    def to_array(self) -> bc.xp.ndarray:
        """Serialize the history buffers to a 1D array."""
        ...

    @classmethod
    def from_array(cls, array: bc.xp.ndarray) -> Self:
        """Deserialize a 1D array back into the history buffers."""
        ...
        
    @classmethod
    def generate_random(cls, key: bc.xp.ndarray) -> Self:
        """Generate a random observation history."""
        ...
        
    @classmethod
    def generate_zero(cls) -> Self:
        """Generate a zero observation history."""
        ...
    
    
def make_history_cls(base_class: Type[T], lengths: Dict[str, int], delay: int) -> Type[ObservationHistory]:
    __FIELD_SPECS__ = {}
    
    for name, anno in base_class.__annotations__.items():
            field_type = anno.__origin__
            field_shape = anno.__metadata__[0]
            hist_size = lengths.get(name, 1)  # Default to 1 if not specified
            __FIELD_SPECS__[name] = (hist_size, field_shape)
            
    @classmethod
    def init(cls, example: T):
        """Initialize the history buffers with specified lengths."""
        assert base_class._validate(example)  # Validate the example instance 
        out = {}
        for f in fields(base_class):
            name = f.name
            hist_t = HistBuf[get_type_hints(base_class)[name]]
            length = lengths.get(name, 1)  # Default to 1 if not specified
            example_value = getattr(example, name)
            out[name] = hist_t.init(example_value, length, delay)
            
        return cls(**out)
    
    @classmethod
    def _push(cls, obs_hist, inst: T):
        """Push an instance of the base class into the history buffers."""
        assert base_class._validate(inst)  # Validate the instance being pushed
        for name in get_type_hints(base_class):
            new_history_buf = getattr(obs_hist, name).push(getattr(inst, name))
            obs_hist = obs_hist.replace(**{name: new_history_buf})
        return obs_hist
    
    def push(self, inst: T):
        """Push an instance of the base class into the history buffers."""
        return self._push(self, inst)

    def to_array(self):
        """Serializes the history buffers to a 1D array."""
        arrays = []
        for field_name in __FIELD_SPECS__:
            field_data = getattr(self, field_name).stack().ravel()
            arrays.append(field_data)
        return bc.xp.concatenate(arrays)

    @classmethod
    def from_array(cls, array: bc.xp.ndarray):
        """Deserializes a 1D array back into the history buffers."""
        raise NotImplementedError("from_array is not implemented for this class.")
        out = {}
        idx = 0
        for field_name, (length, shape) in __FIELD_SPECS__.items():
            field_size = math.prod(shape)
            field_data = array[idx:idx + length * field_size].reshape((length, *shape))
            out[field_name] = HistBuf.init(field_data[0], length, 0)
            for i in range(1, length):
                out[field_name] = out[field_name].push(field_data[i])
            idx += length * field_size
        return cls(**out)
    
    @classmethod
    def generate_random(cls, key: bc.xp.ndarray):
        """Generate a random observation history."""
        max_hist = max(1, max(lengths.values()))
        # Initialize
        random_obs = base_class.generate_random(bc.random.fold_in(key, 0))
        obs_hist = cls.init(random_obs)
        
        for i in range(max_hist):
            random_obs = base_class.generate_random(bc.random.fold_in(key, i+1))
            obs_hist = obs_hist.push(random_obs)
                
        return obs_hist
    
    @classmethod
    def generate_zero(cls):
        """Generate a random observation history."""
        # Initialize
        random_obs = base_class.generate_zero()
        obs_hist = cls.init(random_obs)
                
        return obs_hist
        
    
    dc = bc.dataclass(
        type(f"{base_class.__name__}History", (object,), {
            "__annotations__": {name: HistBuf[get_type_hints(base_class)[name]] for name in get_type_hints(base_class)},
            "__FIELD_SPECS__": __FIELD_SPECS__,
            "__BASE_CLASS__": base_class,
            "init": init,
            "push": push,
            "_push": _push,
            "to_array": to_array,
            "from_array": from_array,
            "generate_random": generate_random,
            "generate_zero":generate_zero
        })
    )
    
    dc.ObservationBase = base_class
    return dc


if __name__ == "__main__":
    from observation import DroneViconObs

    # Example usage
    lengths = {
        "drone_position": 2,
        "drone_velocity": 3,
        "drone_body_rate": 2
    }
    
    DroneViconObsHistory = make_history_cls(DroneViconObs, lengths, delay=2)
    
    key = bc.random.PRNGKey(0)
    print("key:", key)
    obs_hist = DroneViconObsHistory.generate_random(key)
    
    print(obs_hist.drone_position)
    print(obs_hist.drone_velocity.stack())
    print(obs_hist.drone_body_rate.stack())
