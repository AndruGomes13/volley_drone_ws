from typing import Optional, Type

from dataclasses import dataclass
from bounce_policy_pkg.observation.observation import DroneViconObs, DroneBallRelativeViconObs, Observation, ObservationType
from bounce_policy_pkg.observation.observation_data import ObservationData
from bounce_policy_pkg.types.ball import BallState
from bounce_policy_pkg.types.drone import DroneState
import numpy as np

def populate_observation(observation_class: Type[Observation], data: ObservationData) -> Observation:
    """
    Populates the observation in the given state.
    """
    if observation_class is DroneViconObs:
        assert data.drone_state is not None, "Drone state must be provided for DroneViconObs"
        assert data.last_policy_request is not None, "Last policy request must be provided for DroneViconObs"
        return observation_class(drone_position=data.drone_state.position - np.array([0,0,1]),
                                drone_orientation=data.drone_state.orientation_wxyz,
                                drone_velocity=data.drone_state.velocity,
                                drone_body_rate=data.drone_state.body_rate,
                                previous_action=data.last_policy_request,
                                )
    if observation_class is DroneBallRelativeViconObs:
        assert data.drone_state is not None, "Drone state must be provided for DroneBallRelativeViconObs"
        assert data.ball_state is not None, "Ball state must be provided for DroneBallRelativeViconObs"
        assert data.last_policy_request is not None, "Last policy request must be provided for DroneBallRelativeViconObs"
        MIN_CLIP = np.array((-1.5, -1.5, -1))
        MAX_CLIP = np.array((1.5, 1.5, 2.5))
        ball_relative_to_drone_clipped = np.clip(data.ball_state.position - data.drone_state.position, MIN_CLIP, MAX_CLIP)
        return observation_class(drone_position=data.drone_state.position,
                                 drone_orientation=data.drone_state.orientation_wxyz,
                                 drone_velocity=data.drone_state.velocity,
                                 drone_body_rate=data.drone_state.body_rate,
                                 previous_action=data.last_policy_request,
                                 ball_velocity=data.ball_state.velocity,
                                 ball_relative_to_drone=ball_relative_to_drone_clipped,
                                 )
   
    else:
        raise ValueError(f"Unsupported observation class: {observation_class}")
        
        
def get_observation_class(observation_type: ObservationType) -> Type[Observation]:
    """
    Returns the observation class based on the observation type.
    """
    if observation_type == ObservationType.DRONE_VICON:
        return DroneViconObs
    elif observation_type == ObservationType.DRONE_BALL_RELATIVE_VICON:
        return DroneBallRelativeViconObs
    else:
        raise ValueError(f"Unknown observation type: {observation_type}")