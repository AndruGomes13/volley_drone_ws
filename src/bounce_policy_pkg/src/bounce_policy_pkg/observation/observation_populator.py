from typing import Optional, Type

from dataclasses import dataclass
from bounce_policy_pkg.observation.observation import DroneViconObs, DroneBallRelativeViconObs, Observation, ObservationType
from bounce_policy_pkg.observation.observation_data import ObservationData
from bounce_policy_pkg.types.ball import BallState
from bounce_policy_pkg.types.drone import DroneState
import numpy as np
from scipy.spatial.transform import Rotation as R

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
    # if observation_class is DroneBallRelativeViconObs:
    #     assert data.drone_state is not None, "Drone state must be provided for DroneBallRelativeViconObs"
    #     assert data.ball_state is not None, "Ball state must be provided for DroneBallRelativeViconObs"
    #     assert data.last_policy_request is not None, "Last policy request must be provided for DroneBallRelativeViconObs"
    #     USE_BOX_CLIP = False
    #     if USE_BOX_CLIP:
    #         MIN_CLIP = np.array((-1.5, -1.5, -1))
    #         MAX_CLIP = np.array((1.5, 1.5, 2.5))
    #         ball_relative_to_drone_clipped = np.clip(data.ball_state.position - data.drone_state.position, MIN_CLIP, MAX_CLIP)

    #     if not USE_BOX_CLIP:
    #         XY_CLIP_DIST = 1.5
    #         Z_CLIP = (-0.5, 2.5)
    #         ball_relative_to_drone = data.ball_state.position - data.drone_state.position
    #         xy_relative_dist = np.linalg.norm(ball_relative_to_drone[:2])
    #         xy_relative_dist_clipped = np.clip(xy_relative_dist, 0, XY_CLIP_DIST)
    #         xy_relative_ori = ball_relative_to_drone[:2] / (xy_relative_dist + 1e-6)  # Avoid division by zero
    #         z_relative = np.clip(ball_relative_to_drone[2], Z_CLIP[0], Z_CLIP[1])
    #         ball_relative_to_drone_clipped = np.array([xy_relative_ori[0] * xy_relative_dist_clipped, 
    #                                                 xy_relative_ori[1] * xy_relative_dist_clipped, 
    #                                                 z_relative])
    #     return observation_class(drone_position=data.drone_state.position,
    #                              drone_orientation=data.drone_state.orientation_wxyz,
    #                              drone_velocity=data.drone_state.velocity,
    #                              drone_body_rate=data.drone_state.body_rate,
    #                              previous_action=data.last_policy_request,
    #                              ball_velocity=data.ball_state.velocity,
    #                              ball_relative_to_drone=ball_relative_to_drone_clipped,
    #                              )
   
    elif observation_class is DroneBallRelativeViconObs:
        assert data.drone_state is not None, "Drone state must be provided for DroneBallRelativeViconObs"
        assert data.ball_state is not None, "Ball state must be provided for DroneBallRelativeViconObs"
        assert data.last_policy_request is not None, "Last policy request must be provided for DroneBallRelativeViconObs"
        
        # --- Relative ball clipping ---
        USE_BOX_CLIP = False
        if USE_BOX_CLIP:
            MIN_CLIP = np.array((-1.5, -1.5, -1))
            MAX_CLIP = np.array((1.5, 1.5, 2.5))
            ball_relative_to_drone_clipped = np.clip(data.ball_state.position - data.drone_state.position, MIN_CLIP, MAX_CLIP)

        if not USE_BOX_CLIP:
            XY_CLIP_DIST = 1.5
            Z_CLIP = (-0.5, 2.5)
            ball_relative_to_drone = data.ball_state.position - data.drone_state.position
            xy_relative_dist = np.linalg.norm(ball_relative_to_drone[:2])
            xy_relative_dist_clipped = np.clip(xy_relative_dist, 0, XY_CLIP_DIST)
            xy_relative_ori = ball_relative_to_drone[:2] / (xy_relative_dist + 1e-6)  # Avoid division by zero
            z_relative = np.clip(ball_relative_to_drone[2], Z_CLIP[0], Z_CLIP[1])
            ball_relative_to_drone_clipped = np.array([xy_relative_ori[0] * xy_relative_dist_clipped, 
                                                    xy_relative_ori[1] * xy_relative_dist_clipped, 
                                                    z_relative])
            
        # --- Orientation and yaw ---
        quat_wxyz = data.drone_state.orientation_wxyz
        rot = R.from_quat(quat_wxyz[np.array([1, 2, 3, 0])])  # Convert to R.from_quat format
        g_vec = rot.inv().apply(np.array([0, 0, -1]))
        yaw, pitch, roll = rot.as_euler('ZYX', degrees=False)
        sin_yaw = np.sin(yaw).reshape((1,))
        cos_yaw = np.cos(yaw).reshape((1,))

        obs = observation_class(drone_position=data.drone_state.position,
                                    g_vec=g_vec,
                                    sin_yaw=sin_yaw,
                                    cos_yaw=cos_yaw, 
                                 drone_velocity=data.drone_state.velocity,
                                 drone_body_rate=data.drone_state.body_rate,
                                 previous_action=data.last_policy_request,
                                 ball_velocity=data.ball_state.velocity,
                                 ball_relative_to_drone=ball_relative_to_drone_clipped,
                                 )
        return obs
   
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