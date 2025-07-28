#include "Eigen/src/Geometry/Quaternion.h"
#include "geometry_msgs/PoseStamped.h"
#include "motion_capture_ros_msgs/PointCloud.h"
#include "ros/console.h"
#include "ros/time.h"
#include <ros/ros.h>
#include <libmotioncapture/motioncapture.h>
#include <string>
#include <unordered_map>



class MotionCaptureNode
{
    public:
        MotionCaptureNode(ros::NodeHandle &nh, ros::NodeHandle &private_nh):
            nh_(nh),
            pnh_(private_nh)
        {
            ROS_INFO("Motion Capture Node initialized.");
            
            // Load parameters
            std::string motionCaptureHostname = "";
            private_nh.getParam("type", motionCaptureType);
            private_nh.getParam("hostname", motionCaptureHostname);
            private_nh.getParam("interface_ip", motionCaptureInterfaceIP);
            private_nh.getParam("port_command", motionCapturePortCommand);

            // Populate configuration map
            motionCaptureCfg["hostname"] = motionCaptureHostname;
            motionCaptureCfg["type"] = motionCaptureType;
            motionCaptureCfg["interface_ip"] = motionCaptureInterfaceIP;
            // motionCaptureCfg["port_command"] = std::to_string(motionCapturePortCommand);

            mocap_ = std::unique_ptr<libmotioncapture::MotionCapture>(
               libmotioncapture::MotionCapture::connect(motionCaptureType,
                                                        motionCaptureCfg));
            if (!mocap_)
            {
                ROS_FATAL("Failed to connect to motion capture system of type: %s", motionCaptureType.c_str());
                ros::shutdown();
                return;
            }
            else
            {
                ROS_INFO("Connected to motion capture system of type: %s at %s", motionCaptureType.c_str(), motionCaptureHostname.c_str());
            }

            // Create publishers
            pointCloudMotionCapturePub = nh.advertise<motion_capture_ros_msgs::PointCloud>("mocap/unlabeled_point_cloud", 1);

            ROS_INFO("Point cloud publisher created.");
        }

        ~MotionCaptureNode()
        {
            ROS_INFO("Motion Capture Node shutting down.");
        }

        ros::Publisher& publisher_from_body(const std::string& body_name)
        {
            auto it = body_to_publisher_map.find(body_name);
            if (it == body_to_publisher_map.end()) {
                ros::Publisher pub = nh_.advertise<geometry_msgs::PoseStamped>("mocap/"+body_name+"/pose", 1);
                it = body_to_publisher_map.emplace(body_name, pub).first;
            }
            return it->second;
        }

        void tick()
        {
            mocap_->waitForNextFrame();
            auto time = ros::Time::now();
            
            // Compensate latency
            auto latency = mocap_->latency();
            float total_latency = 0;
            for (const auto &iter : latency) {
                total_latency += iter.value();
            }
            time -= ros::Duration(total_latency);
            
            // Publish body poses
            ROS_INFO("Publishing poses for %zu bodies", mocap_->rigidBodies().size());
            for (const auto &iter : mocap_->rigidBodies()){
                std::cout << "Publishing pose for body: " << iter.first << std::endl;
                publishBodyPose(iter.second, time);
            }

            publishPointCloud(mocap_->pointCloud(), time);
        }

    private:
        void publishBodyPose(const libmotioncapture::RigidBody& rigidBody, 
                             const ros::Time& time)
        {

            static const Eigen::Quaternionf q_fix(
                std::cos(M_PI/4),   // w  = √½
                std::sin(M_PI/4),   // x  = √½   (+90° about +X)
                0, 0);              // y,z = 0
            
            
            Eigen::Quaterniond q_ros;
            if (motionCaptureType=="optitrack") {
                q_ros=rigidBody.rotation() * q_fix;
            } else {
                q_ros=rigidBody.rotation();
            }
            

            auto& pub = publisher_from_body(rigidBody.name());
            geometry_msgs::PoseStamped poseMsg;
            poseMsg.header.stamp = time;
            poseMsg.header.frame_id = "world";  
            poseMsg.pose.position.x = rigidBody.position().x();
            poseMsg.pose.position.y = rigidBody.position().y();
            poseMsg.pose.position.z = rigidBody.position().z();
            poseMsg.pose.orientation.x = q_ros.x();
            poseMsg.pose.orientation.y = q_ros.y();
            poseMsg.pose.orientation.z = q_ros.z();
            poseMsg.pose.orientation.w = q_ros.w();
            pub.publish(poseMsg);

        }

        void publishPointCloud(const libmotioncapture::PointCloud& point_cloud, const ros::Time& time)
        {
            int num_points = point_cloud.rows();
            std::cout << "Publishing point cloud with " << num_points << " points." << std::endl;
            motion_capture_ros_msgs::PointCloud point_cloud_msg;

            point_cloud_msg.t = time.toSec();

            for (int i = 0; i < num_points; ++i) {

                geometry_msgs::PoseStamped pointPoseMsg;
                pointPoseMsg.header.stamp = time;
                pointPoseMsg.header.frame_id = "world";
                pointPoseMsg.pose.position.x = point_cloud(i, 0);
                pointPoseMsg.pose.position.y = point_cloud(i, 1);
                pointPoseMsg.pose.position.z = point_cloud(i, 2);
                pointPoseMsg.pose.orientation.w = 1.0; // Assuming no rotation for points
                point_cloud_msg.poses.push_back(pointPoseMsg);
            }
            pointCloudMotionCapturePub.publish(point_cloud_msg);
        }

        ros::NodeHandle nh_;
        ros::NodeHandle pnh_;

        std::map<std::string, std::string> motionCaptureCfg;
        std::string motionCaptureHostname;
        std::string motionCaptureType;
        std::string motionCaptureInterfaceIP;
        int motionCapturePortCommand;

        ros::Publisher bodyMotionCapturePub;
        ros::Publisher pointCloudMotionCapturePub;

        std::unordered_map<std::string, ros::Publisher> body_to_publisher_map;
        std::unique_ptr<libmotioncapture::MotionCapture> mocap_;
};



int main(int argc, char **argv)
{
  ros::init(argc, argv, "motion_capture_node");
  ros::NodeHandle nh, private_nh("~");

    MotionCaptureNode motionCaptureNode(nh, private_nh);
    while (ros::ok())
    {
        motionCaptureNode.tick();
        ros::spinOnce();
    
    }
    
  return 0;
}