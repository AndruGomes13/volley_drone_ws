from typing import Optional, Type

from dataclasses import dataclass
from policy_package.observation.observation import DroneStateObs, DroneViconObs, FullObservationWithViconFlag, FullStateObservation, Observation, ObservationType
from policy_package.observation.observation_data import ObservationData
from policy_package.types.ball import BallState
from policy_package.types.drone import DroneState
import numpy as np

def populate_observation(observation_class: Type[Observation], data: ObservationData) -> Observation:
    """
    Populates the observation in the given state.
    """
    if observation_class is DroneViconObs:
        assert data.drone_state is not None, "Drone state must be provided for DroneViconObs"
        assert data.last_policy_request is not None, "Last policy request must be provided for DroneViconObs"
        return observation_class(drone_position=data.drone_state.position,
                                drone_orientation=data.drone_state.orientation_wxyz,
                                drone_velocity=data.drone_state.velocity,
                                drone_body_rate=data.drone_state.body_rate,
                                previous_action=data.last_policy_request,
                                )
    else:
        raise ValueError(f"Unsupported observation class: {observation_class}")
        
        
def get_observation_class(observation_type: ObservationType) -> Type[Observation]:
    """
    Returns the observation class based on the observation type.
    """
    if observation_type == ObservationType.DRONE_STATE:
        return DroneStateObs
    elif observation_type == ObservationType.DRONE_VICON:
        return DroneViconObs
    elif observation_type == ObservationType.FULL_OBSERVATION_WITH_VICON_FLAG:
        return FullObservationWithViconFlag
    elif observation_type == ObservationType.FULL_STATE_OBSERVATION:
        return FullStateObservation
    else:
        raise ValueError(f"Unknown observation type: {observation_type}")