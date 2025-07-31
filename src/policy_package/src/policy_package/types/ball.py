from dataclasses import dataclass
import numpy as np
from typing_extensions import Self
from agiros_msgs.msg._BallState import BallState as BallStateMsg


@dataclass
class BallState:
    time:float
    position: np.ndarray
    velocity: np.ndarray
    
    #TODO
    @classmethod
    def from_msg(cls, msg: BallStateMsg) -> Self: 
        
        time = msg.header.stamp.to_sec()
        position = np.array([msg.position.x, msg.position.y, msg.position.z])
        velocity = np.array([msg.linear_velocity.x, msg.linear_velocity.y, msg.linear_velocity.z])

        return cls(
            time=time,
            position=position,
            velocity=velocity,
        )
    @classmethod
    def generate_random(cls) -> Self:
        return cls(
            time=np.random.uniform(0, 10),
            position=np.random.uniform(-10, 10, size=3),
            velocity=np.random.uniform(-1, 1, size=3)
        )