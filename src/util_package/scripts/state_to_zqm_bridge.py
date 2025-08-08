#!/usr/bin/env python3
from dataclasses import dataclass
import numpy as np
import zmq
import rospy
from agiros_msgs.msg import QuadState

@dataclass
class StateEstimate:
    position: np.ndarray
    velocity: np.ndarray
    orientation: np.ndarray
    angular_velocity: np.ndarray

    def to_dict(self):
        return {
            "position": self.position.tolist(),
            "velocity": self.velocity.tolist(),
            "orientation": self.orientation.tolist(),
            "angular_velocity": self.angular_velocity.tolist(),
        }

    @staticmethod
    def from_dict(d):
        return StateEstimate(
            position=np.array(d["position"]),
            velocity=np.array(d["velocity"]),
            orientation=np.array(d["orientation"]),
            angular_velocity=np.array(d["angular_velocity"]),
        )


# TODO: Add this to a ROS node that publishes state estimates.
class StateEstimatePublisher:
    """ Will use ZQM to publish state estimates to the ROS node."""
    def __init__(self):
        PORT = 5555
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.PUB)
        self.socket.bind(f"tcp://*:{PORT}")
        
    def publish_state_estimate(self, state_estimate: StateEstimate):
        """Publish a state estimate to the server."""
        message = {
            "topic": "/drone_state_estimate",
            "data": state_estimate.to_dict()
        }
        self.socket.send_json(message)

def get_state_estimate_callback(publisher: StateEstimatePublisher):
    def state_estimate_callback(msg: QuadState):
        """Callback function to convert QuadState to StateEstimate and publish."""
        state_estimate = StateEstimate(
            position=np.array([msg.pose.position.x, msg.pose.position.y, msg.pose.position.z]),
            velocity=np.array([msg.velocity.linear.x, msg.velocity.linear.y, msg.velocity.linear.z]),
            orientation=np.array([msg.pose.orientation.x, msg.pose.orientation.y, msg.pose.orientation.z, msg.pose.orientation.w]),
            angular_velocity=np.array([msg.velocity.angular.x, msg.velocity.angular.y, msg.velocity.angular.z])
        )
        publisher.publish_state_estimate(state_estimate)

    return state_estimate_callback

if __name__ == "__main__":
    rospy.init_node("state_estimate_ros_zmq_bridge")
    quad_name = rospy.get_param("~quad_name", "volley_drone")
    publisher = StateEstimatePublisher()
    state_estimate_callback = get_state_estimate_callback(publisher)
    state_estimate_sub = rospy.Subscriber(f"/{quad_name}/agiros_pilot/state", QuadState, state_estimate_callback)
    rospy.spin()  # Keep the node running to listen for messages
    publisher.socket.close()
    publisher.context.term()  # Clean up ZMQ context
    rospy.loginfo("State estimate publisher node has been shut down.")
    