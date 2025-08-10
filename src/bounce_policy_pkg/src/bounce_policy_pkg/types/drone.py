from dataclasses import dataclass
import numpy as np
from typing_extensions import Self
from agiros_msgs.msg._QuadState import QuadState


@dataclass
class DroneState:
    time:float
    position: np.ndarray
    velocity: np.ndarray
    orientation_wxyz: np.ndarray
    body_rate: np.ndarray
    angular_acceleration: np.ndarray
    gyro_bias: np.ndarray
    
    @classmethod
    def from_msg(cls, msg:QuadState) -> Self:
        time = msg.header.stamp.to_sec()
        position = np.array([msg.pose.position.x, msg.pose.position.y, msg.pose.position.z])
        velocity = np.array([msg.velocity.linear.x, msg.velocity.linear.y, msg.velocity.linear.z])
        orientation_wxyz = np.array([msg.pose.orientation.w, msg.pose.orientation.x, msg.pose.orientation.y, msg.pose.orientation.z])
        body_rate = np.array([msg.velocity.angular.x, msg.velocity.angular.y, msg.velocity.angular.z])
        angular_acceleration = np.array([msg.acceleration.angular.x, msg.acceleration.angular.y, msg.acceleration.angular.z])
        gyro_bias = np.array([msg.gyr_bias.x, msg.gyr_bias.y, msg.gyr_bias.z])
        

        return cls(
            time=time,
            position=position,
            velocity=velocity,
            orientation_wxyz=orientation_wxyz,
            body_rate=body_rate,
            angular_acceleration=angular_acceleration,
            gyro_bias=gyro_bias
        )
    @classmethod
    def generate_random(cls) -> Self:
        return cls(
            time=np.random.uniform(0, 10),
            position=np.random.uniform(-10, 10, size=3),
            velocity=np.random.uniform(-1, 1, size=3),
            orientation_wxyz=np.random.uniform(-1, 1, size=4),
            body_rate=np.random.uniform(-1, 1, size=3),
            angular_acceleration=np.random.uniform(-1, 1, size=3),
            gyro_bias=np.random.uniform(-0.1, 0.1, size=3)
        )