from dataclasses import dataclass
from typing import Optional

from policy_package.types.ball import BallState
from policy_package.types.drone import DroneState
import numpy as np

@dataclass
class ObservationData:
    """
    A class to hold data for observations.
    """
    drone_state: Optional[DroneState]
    ball_state: Optional[BallState]
    last_policy_request: Optional[np.ndarray]