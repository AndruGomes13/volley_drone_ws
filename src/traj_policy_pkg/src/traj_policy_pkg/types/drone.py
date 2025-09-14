from dataclasses import dataclass
import numpy as np
from typing_extensions import Self
from agiros_msgs.msg._QuadState import QuadState
from sim_types.DroneState import DroneState as _BaseDroneState
from sim_utils.jax_numpy_backend import numpy_jax_backend as bc


@bc.dataclass
class DroneState(_BaseDroneState):    
    @classmethod
    def from_msg(cls, msg:QuadState) -> Self:
        time = msg.header.stamp.to_sec()
        position = np.array([msg.pose.position.x, msg.pose.position.y, msg.pose.position.z])
        velocity = np.array([msg.velocity.linear.x, msg.velocity.linear.y, msg.velocity.linear.z])
        orientation_wxyz = np.array([msg.pose.orientation.w, msg.pose.orientation.x, msg.pose.orientation.y, msg.pose.orientation.z])
        body_rate = np.array([msg.velocity.angular.x, msg.velocity.angular.y, msg.velocity.angular.z])
        

        return cls(
            time=time,
            position=position,
            velocity=velocity,
            orientation_wxyz=orientation_wxyz,
            body_rate=body_rate,
        )
    @classmethod
    def generate_random(cls) -> Self:
        return cls(
            time=np.random.uniform(0, 10),
            position=np.random.uniform(-10, 10, size=3),
            velocity=np.random.uniform(-1, 1, size=3),
            orientation_wxyz=np.random.uniform(-1, 1, size=4),
            body_rate=np.random.uniform(-1, 1, size=3),
        )