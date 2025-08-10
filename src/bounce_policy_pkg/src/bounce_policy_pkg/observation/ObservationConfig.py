

from bounce_policy_pkg.observation.observation import ObservationType
from pydantic import BaseModel
from typing import Dict


class ObservationConfig(BaseModel):
    """
    Configuration class for observation settings in the drone environment.
    This class defines various observation parameters and their default values.
    """
    # --- Actor ---
    
    actor_observation_type: ObservationType = ObservationType.DRONE_VICON
    history_length_actor: Dict[str, int] = {
            "drone_position": 4,
            "drone_orientation": 4,
            "drone_velocity": 4,
            "drone_body_rate": 4,
            "previous_action": 4,
        }
    
    
    