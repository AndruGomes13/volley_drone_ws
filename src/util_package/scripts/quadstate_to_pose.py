#!/usr/bin/env python
import rospy
from agiros_msgs.msg import QuadState
from geometry_msgs.msg import PoseStamped

def callback(msg):
    pose_msg = PoseStamped()
    pose_msg.header = msg.header
    pose_msg.pose = msg.pose
    pub.publish(pose_msg)

rospy.init_node('quadstate_to_pose')
sub = rospy.Subscriber('/volley_drone/agiros_pilot/state', QuadState, callback)
pub = rospy.Publisher('/volley_drone/agiros_pilot/state/pose', PoseStamped, queue_size=10)
rospy.spin()
