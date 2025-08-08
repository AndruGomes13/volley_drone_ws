#!/usr/bin/env python3
from dataclasses import dataclass
import numpy as np
import zmq
import rospy
from agiros_msgs.msg import QuadState
from motion_capture_ros_msgs.msg import PointCloud
from geometry_msgs.msg import PointStamped
@dataclass
class StateEstimate:
    position: np.ndarray
    velocity: np.ndarray
    orientation_wxyz: np.ndarray
    angular_velocity: np.ndarray

    def to_dict(self):
        return {
            "position": self.position.tolist(),
            "velocity": self.velocity.tolist(),
            "orientation": self.orientation_wxyz.tolist(),
            "angular_velocity": self.angular_velocity.tolist(),
        }

    @staticmethod
    def from_dict(d):
        return StateEstimate(
            position=np.array(d["position"]),
            velocity=np.array(d["velocity"]),
            orientation_wxyz=np.array(d["orientation"]),
            angular_velocity=np.array(d["angular_velocity"]),
        )

@dataclass
class Point:
    position: np.ndarray

    def to_dict(self):
        return {
            "position": self.position.tolist(),
        }
    @staticmethod
    def from_dict(d):
        return Point(
            position=np.array(d["position"]),
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


class MotionCapturePointCloudClient:
    """ Will use ZQM to receive the point cloud data from the ROS node."""
    def __init__(self):
        PORT = 5556
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.SUB)
        self.socket.connect(f"tcp://localhost:{PORT}")
        self.socket.setsockopt_string(zmq.SUBSCRIBE, "")
        
    def receive_point_cloud(self) -> list[Point]:
        """Receive a point cloud from the server."""
        try:
            message = self.socket.recv_json(flags=zmq.NOBLOCK)
            if message["topic"] == "/mocap_point_cloud":
                return [Point.from_dict(point) for point in message["data"]]
        except zmq.Again:
            return []
        except Exception as e:
            print(f"Error receiving point cloud: {e}")
            return []

def get_state_estimate_callback(publisher: StateEstimatePublisher):
    def state_estimate_callback(msg: QuadState):
        """Callback function to convert QuadState to StateEstimate and publish."""
        state_estimate = StateEstimate(
            position=np.array([msg.pose.position.x, msg.pose.position.y, msg.pose.position.z]),
            velocity=np.array([msg.velocity.linear.x, msg.velocity.linear.y, msg.velocity.linear.z]),
            orientation_wxyz=np.array([msg.pose.orientation.w, msg.pose.orientation.x, msg.pose.orientation.y, msg.pose.orientation.z]),
            angular_velocity=np.array([msg.velocity.angular.x, msg.velocity.angular.y, msg.velocity.angular.z])
        )
        publisher.publish_state_estimate(state_estimate)

    return state_estimate_callback

def get_point_cloud_pub_callback(client: MotionCapturePointCloudClient, publisher: rospy.Publisher):
    def point_cloud_pub_callback(event):
        """Callback function to convert PointCloud to list of Points."""
        point_cloud: list[Point] = client.receive_point_cloud()
        if point_cloud is None:
            return
        
        msg = PointCloud()
        t = rospy.Time.now()
        msg.t = t.to_sec()
        
        for point in point_cloud:
            point_msg = PointStamped()
            point_msg.point.x = point.position[0]
            point_msg.point.y = point.position[1]
            point_msg.point.z = point.position[2]
            point_msg.header.stamp = t
            point_msg.header.frame_id = "world"  # Set the frame_id as

            msg.points.append(point_msg)

        publisher.publish(msg)
    return point_cloud_pub_callback


POINT_CLOUD_HZ = 500
if __name__ == "__main__":
    rospy.init_node("state_estimate_ros_zmq_bridge")
    quad_name = rospy.get_param("~quad_name", "volley_drone")
    
    
    
    publisher = StateEstimatePublisher()
    state_estimate_callback = get_state_estimate_callback(publisher)
    state_estimate_sub = rospy.Subscriber(f"/{quad_name}/agiros_pilot/state", QuadState, state_estimate_callback)

    point_cloud_client = MotionCapturePointCloudClient()
    state_estimate_pub = rospy.Publisher(f"/mocap/unlabeled_point_cloud", PointCloud, queue_size=10)
    point_cloud_pub_callback = get_point_cloud_pub_callback(point_cloud_client, state_estimate_pub)
    rospy.Timer(rospy.Duration.from_sec(1/POINT_CLOUD_HZ), point_cloud_pub_callback)

    rospy.spin()  # Keep the node running to listen for messages
    publisher.socket.close()
    publisher.context.term()  # Clean up ZMQ context
    rospy.loginfo("State estimate publisher node has been shut down.")
    