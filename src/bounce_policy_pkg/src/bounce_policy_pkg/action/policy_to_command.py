

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Tuple, Optional
from pydantic import BaseModel
import numpy as np
import rospy

from agiros_msgs.msg._Command import Command
from geometry_msgs.msg import Vector3

G = 9.81  # Gravitational constant
DEG = np.pi / 180  # Degrees to radians conversion
Tuple3 = Tuple[float, float, float]
Tuple4 = Tuple[float, float, float, float]

# --- Config ----
class ActionModelConfig(BaseModel):    
    # --- Drone parameters ---
    thrust_to_weight_ratio: float = 4.0
    
    # --- Policy to Command ---
    max_body_rate: Tuple3 = (1022*DEG, 1022*DEG, 524*DEG) # Shape (3,) for roll, pitch, yaw
    max_command_rate_change: Tuple4 = (4*G/100, 20*DEG, 20*DEG, 20*DEG)
    use_command_rate_change_clipping: bool = False

class PolicyToCommandMapper(ABC):
    def __init__(self, config: ActionModelConfig):
        self.config = config  
    
    @abstractmethod
    def map(self, policy_action: np.ndarray, time : Optional[rospy.Time] = None) -> Command:
        """
        Map the policy action request to the command space.
        - policy_action_request should be [-1,1] normalized and size 4
        """
        pass

class PolicyToNormalizedThrustAndBodyRate(PolicyToCommandMapper):
    '''
    This class maps the policy action request to a normalized thrust and body rate command.
    The policy action request is expected to be in the range [-1, 1].
    Effectively, it maps the policy action request to an acceleration (mass agnostic) and angular velocity command.
    
    '''
    def __init__(self, config: ActionModelConfig):
        super().__init__(config)
        self.COMMAND_LOWER_BOUND = np.array([
            0.0,
            -config.max_body_rate[0],
            -config.max_body_rate[1],
            -config.max_body_rate[2]    
        ])
        self.COMMAND_UPPER_BOUND = np.array([
            config.thrust_to_weight_ratio * G,
            config.max_body_rate[0],
            config.max_body_rate[1],
            config.max_body_rate[2]
        ])

    def map(self, policy_action: np.ndarray, time : Optional[rospy.Time] = None) -> Command:
        """Map the policy action to normalized thrust and body rate command."""
        assert policy_action.shape == (4,), f"Expected policy action shape (4,), got {policy_action.shape}"
        thrust = (policy_action[0] + 1.0) * 0.5 * (self.COMMAND_UPPER_BOUND[0] - self.COMMAND_LOWER_BOUND[0]) + self.COMMAND_LOWER_BOUND[0]
        body_rate = (policy_action[1:] + 1.0) * 0.5 * (self.COMMAND_UPPER_BOUND[1:] - self.COMMAND_LOWER_BOUND[1:]) + self.COMMAND_LOWER_BOUND[1:]

        return Command(
            t=time if time is not None else 0.0,
            is_single_rotor_thrust=False,  # Assuming collective thrust and body rates
            collective_thrust=thrust,
            bodyrates=Vector3(body_rate[0], body_rate[1], body_rate[2]),
        )
