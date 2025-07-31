#!/usr/bin/env python3
import datetime
import os
from pathlib import Path
from typing import Union

import numpy as np
import pandas as pd
import rospy
import torch
from agiros_msgs.msg import Command, QuadState, Telemetry
from flightgym import ObstacleAvoidanceVecEnv_v0
from geometry_msgs.msg import Point
from rpg_baselines.torch.envs import vec_env_obs_wrapper as wrapper
from ruamel.yaml import YAML, RoundTripDumper, dump
from scipy.spatial.transform import Rotation as R
from stable_baselines3.common.utils import get_device
from stable_baselines3.ppo.policies import MlpPolicy
from std_msgs.msg import Bool, String
from visualization_msgs.msg import Marker

from src.utils.ev_train_utils import mask_out_lidar_scan
from src.utils.ev_visualizations import visualize_step

 


class ObstacleAvoidancePilot(object):
    def __init__(
        self,
        policy_dir,
        policy: MlpPolicy,
        env: wrapper.ObstacleEnvVec,
        obstacle_poses,
        env_config,
        quad_config,
        device,
    ):

        self.save_dir = policy_dir / "deploy"
        self.policy_dir = policy_dir
        self.iter_num = 0
        self.log_df = pd.DataFrame(
            None,
            columns=[
                "t",
                "px",
                "py",
                "pz",
                "qw",
                "qx",
                "qy",
                "qz",
                "vx",
                "vy",
                "vz",
                "omex",
                "omey",
                "omez",
                "c",
                "wx",
                "wy",
                "wz",
                "ubat",
            ],
        )
        self.log_list = []

        self.lstm_states = None  # only used for recurrent policies
        self.obstacle_poses = obstacle_poses

        self.voltage = 16.4
        self.cmd_thrust = 0
        self.cmd_wx = 0
        self.cmd_wy = 0
        self.cmd_wz = 0

        self.use_sim_time = rospy.get_param("/use_sim_time", False)
        self.quad_name = rospy.get_param("~quad_name")

        self.device = device
        self.act_dim = 4

        self.num_passed_gates = 0

        self.env_config = env_config
        self.env = env

        self.quad_config = quad_config
        self.policy = policy

        self.act_mean_ = np.zeros(shape=(1, self.act_dim))
        self.act_std_ = np.zeros(shape=(1, self.act_dim))
        self.act_max_ = np.zeros(shape=(1, self.act_dim))
        self.act_min_ = np.zeros(shape=(1, self.act_dim))

        self._current_lap = 0
        self._max_laps = 15

        self.control_rate = 50

        self.t0 = 0
        self.init()

        self.start = True

        # odm
        self.odm_sub_ = rospy.Subscriber(
            "agiros_pilot/state", QuadState, self.state_callback, queue_size=1
        )
        # Add new subscriber for state monitoring
        self.monitor_sub_ = rospy.Subscriber(
            "agiros_pilot/state",
            QuadState,
            self._monitor_state_callback,
            queue_size=1,
        )
        # Add new subscriber for state monitoring
        self.monitor_sub_ = rospy.Subscriber(
            "agiros_pilot/state",
            QuadState,
            self._monitor_state_callback,
            queue_size=1,
        )
        # control command
        self.cmd_pub_ = rospy.Publisher(
            "agiros_pilot/feedthrough_command",
            Command,
            queue_size=1,
            tcp_nodelay=True,
        )
        # bat
        self.tel_sub = rospy.Subscriber(
            "agiros_pilot/telemetry",
            Telemetry,
            self.telemetry_callback,
            queue_size=1,
        )

        # start signal
        self.start_sub = rospy.Subscriber(
            "/start_policy", Bool, self.start_callback, queue_size=1
        )

        # Add publisher for visualization markers
        self.viz_pub_ = rospy.Publisher(
            "visualization_marker", Marker, queue_size=10
        )

        # self.laptime_pub = rospy.Publisher("/laptime", String, queue_size=1)

        # # action publising with ros timer
        # self.timer_control_loop = rospy.Timer(
        #     rospy.Duration(1.0 / self.control_rate), self.cmd_pub_callback)

    def init(self):

        self.old_position = np.empty(3)
        self.old_position[:] = np.NaN
        self.obs = None
        self.quad_state = None

        self.start_policy = False
        self.first_sample = True
        self.dt = 0.0

        self.frame_id = 0
        self.cmd_id = 0

        self._current_lap = 0
        self._max_laps = 3

        self.quad_mass = self.quad_config["mass"]
        omega_max = self.quad_config["omega_max"]
        thrust_max = self.quad_config["thrust_max"]
        quad_max_force = 4.0 * thrust_max

        print("---------------------------------------")
        print(f"Quadrotor configuration:")
        print("Mass: ", self.quad_mass)
        print("Omega max: ", omega_max)
        print("Thrust max: ", thrust_max)
        print("---------------------------------------")

        if (
            act_norm := self.env_config["environment"].get(
                "action_normalization", "old"
            )
        ) == "old":
            ### old action normalization
            self.act_mean_[0, :] = np.array(
                [(quad_max_force / (2.0 * self.quad_mass)), 0.0, 0.0, 0.0]
            )
            self.act_std_[0, :] = np.array(
                [
                    (quad_max_force / (2.0 * self.quad_mass)),
                    omega_max[0],
                    omega_max[1],
                    omega_max[2],
                ]
            )
            self.act_max_[0, :] = np.array(
                [
                    quad_max_force / self.quad_mass,
                    omega_max[0],
                    omega_max[1],
                    omega_max[2],
                ]
            )
            self.act_min_[0, :] = np.array(
                [0.0, -omega_max[0], -omega_max[1], -omega_max[2]]
            )
        elif act_norm == "new":
            ### new action normalization
            self.act_mean_[0, :] = np.array([9.81, 0.0, 0.0, 0.0])
            self.act_std_[0, :] = np.array(
                [
                    (quad_max_force / self.quad_mass) - 9.81,
                    omega_max[0],
                    omega_max[1],
                    omega_max[2],
                ]
            )
            self.act_max_[0, :] = np.array(
                [
                    quad_max_force / self.quad_mass,
                    omega_max[0],
                    omega_max[1],
                    omega_max[2],
                ]
            )
            self.act_min_[0, :] = np.array(
                [0.0, -omega_max[0], -omega_max[1], -omega_max[2]]
            )
        else:
            raise ValueError(f"Unknown action normalization: {act_norm}")

        self.pre_act = self.act_mean_[0, :] * 0.0

        self.num_prev_act = 1
        self.pre_act_vec = self.num_prev_act * [self.act_mean_[0, :] * 0.0]
        self.img_offset = self.num_prev_act * 4

        self.all_laptimes = []
        self.all_gate_errors = []

    def start_callback(self, start):

        self.start_policy = start.data
        self.t0 = rospy.Time.now().to_sec()

    def cmd_pub_callback(self, timer=None):

        if self.start_policy and self.obs is not None:
            # run policy
            with torch.no_grad():
                obs = torch.as_tensor(self.obs).to(self.device)
                if self.first_sample:
                    obs = 0 * obs
                    self.first_sample = False
                action, self.lstm_states = self.policy.predict(
                    obs, self.lstm_states, deterministic=True
                )

            action = (action * self.act_std_ + self.act_mean_)[0, :]
            action = np.clip(action, self.act_min_[0, :], self.act_max_[0, :])

            self.pre_act = action

            if self.first_sample:
                self.pre_act_vec = self.num_prev_act * [action]
            self.pre_act_vec.append(action)
            self.pre_act_vec.pop(0)

            # publish the command before running the policy
            cmd_msg = Command()
            cmd_msg.header.stamp = rospy.Time.now()
            cmd_msg.t = rospy.Time.now().to_sec()
            cmd_msg.is_single_rotor_thrust = False

            cmd_msg.collective_thrust = self.pre_act[0]
            cmd_msg.bodyrates.x = self.pre_act[1]
            cmd_msg.bodyrates.y = self.pre_act[2]
            cmd_msg.bodyrates.z = self.pre_act[3]

            self.cmd_thrust = self.pre_act[0]
            self.cmd_wx = self.pre_act[1]
            self.cmd_wy = self.pre_act[2]
            self.cmd_wz = self.pre_act[3]

            self.cmd_pub_.publish(cmd_msg)

            self.cmd_id += 1

    def telemetry_callback(self, data):
        if data is not None:
            self.voltage = data.voltage

    def state_callback(self, data: QuadState):
        position = np.array(
            [data.pose.position.x, data.pose.position.y, data.pose.position.z]
        )
        attitude = np.array(
            [
                data.pose.orientation.x,
                data.pose.orientation.y,
                data.pose.orientation.z,
                data.pose.orientation.w,
            ]
        )
        quat = np.array(
            [
                data.pose.orientation.w,
                data.pose.orientation.x,
                data.pose.orientation.y,
                data.pose.orientation.z,
            ]
        )
        velocity = np.array(
            [
                data.velocity.linear.x,
                data.velocity.linear.y,
                data.velocity.linear.z,
            ]
        )
        omega = np.array(
            [
                data.velocity.angular.x,
                data.velocity.angular.y,
                data.velocity.angular.z,
            ]
        )
        acc = np.array(
            [
                data.acceleration.linear.x,
                data.acceleration.linear.y,
                data.acceleration.linear.z,
            ]
        )

        rotation_matrix = (
            R.from_quat(attitude).as_matrix().reshape((9,), order="F")
        )

        # Add markers for visualization
        for i, obstacle in enumerate(self.obstacle_poses[0]):
            marker = Marker()
            marker.header.frame_id = "world"
            marker.header.stamp = rospy.Time.now()
            marker.id = i  # Assign a unique ID to each marker
            marker.type = Marker.CYLINDER  # Set marker type to cylinder
            marker.pose.position.x = obstacle[0]
            marker.pose.position.y = obstacle[1]
            marker.pose.position.z = obstacle[2]
            marker.pose.orientation.x = 0
            marker.pose.orientation.y = 0
            marker.pose.orientation.z = 0.707106781
            marker.pose.orientation.w = 0.707106781
            marker.scale.x = 0.5
            marker.scale.y = 0.5
            marker.scale.z = 2.85
            marker.color.a = 1.0
            marker.color.r = 0.0
            marker.color.g = 0.0
            marker.color.b = 1.0
            self.viz_pub_.publish(marker)

        if self.start_policy:
            self.log_list.append(
                [
                    rospy.Time.now().to_sec(),
                    position[0],
                    position[1],
                    position[2],
                    quat[0],
                    quat[1],
                    quat[2],
                    quat[3],
                    velocity[0],
                    velocity[1],
                    velocity[2],
                    omega[0],
                    omega[1],
                    omega[2],
                    self.cmd_thrust,
                    self.cmd_wx,
                    self.cmd_wy,
                    self.cmd_wz,
                    self.voltage,
                ]
            )

        # * QUADROTOR STATE ----------------------------------------------------
        self.quad_state = (
            np.concatenate(
                [position, quat, velocity, omega, acc, np.zeros(26)], axis=0
            )
            .astype(np.float64)
            .reshape((1, 42))
        )

        self.env.setQuadState(self.quad_state)

        self.old_state = self.quad_state.copy()

        # if self.start:
        #     depth_frame = self.env.get_gt_depth_frames()
        #     self.start = False

        # retrieve observations for control command prediction
        self.obs_time = rospy.Time.now()

        # * INPUT DATA ---------------------------------------------------------
        if self.start_policy:
            # run policy
            with torch.no_grad():
                # odom_data = np.concatenate([velocity, rotation_matrix])
                # odom_data = torch.as_tensor(odom_data).to(self.device)
                # depth_frame = self.env.get_gt_depth_frames()
                # if self.first_sample:
                #     odom_data = 0 * odom_data
                #     self.first_sample = False
                # action, self.lstm_states = self.policy.predict(
                #     vision_data=depth_frame, odometry_data=odom_data)
                obs = self.env.getObs()
                obs = mask_out_lidar_scan(
                    obs, nr_lidar_bins=32, nr_bins_to_mask=22
                )
                action, _ = self.policy.predict(obs, deterministic=True)

            action = (action * self.act_std_ + self.act_mean_)[0, :]
            action = np.clip(action, self.act_min_[0, :], self.act_max_[0, :])

            self.pre_act = action

            if self.first_sample:
                self.pre_act_vec = self.num_prev_act * [action]
            self.pre_act_vec.append(action)
            self.pre_act_vec.pop(0)

            # publish the command before running the policy
            cmd_msg = Command()
            cmd_msg.header.stamp = rospy.Time.now()
            cmd_msg.t = rospy.Time.now().to_sec()
            cmd_msg.is_single_rotor_thrust = False

            cmd_msg.collective_thrust = self.pre_act[0]
            cmd_msg.bodyrates.x = self.pre_act[1]
            cmd_msg.bodyrates.y = self.pre_act[2]
            cmd_msg.bodyrates.z = self.pre_act[3]

            self.cmd_thrust = self.pre_act[0]
            self.cmd_wx = self.pre_act[1]
            self.cmd_wy = self.pre_act[2]
            self.cmd_wz = self.pre_act[3]

            self.cmd_pub_.publish(cmd_msg)

            self.cmd_id += 1
            self.frame_id += 1

        self.old_position = position

    def _monitor_state_callback(self, data: QuadState):
        """
        Monitors the drone's state and stops the policy if it exceeds predefined limits.
        """
        safe_distance = 1
        if not self.start_policy:
            return  # Only monitor if policy is active

        position = np.array(
            [data.pose.position.x, data.pose.position.y, data.pose.position.z]
        )
        attitude = np.array(
            [
                data.pose.orientation.x,
                data.pose.orientation.y,
                data.pose.orientation.z,
                data.pose.orientation.w,
            ]
        )  # x, y, z, w
        velocity = np.array(
            [
                data.velocity.linear.x,
                data.velocity.linear.y,
                data.velocity.linear.z,
            ]
        )
        omega = np.array(
            [
                data.velocity.angular.x,
                data.velocity.angular.y,
                data.velocity.angular.z,
            ]
        )

        # Define thresholds (these are example values and may need tuning)
        # Based on world_box [-6.5, -6.5, 0.7, 13.5, 8.0, 5] from main, slightly tighter
        pos_x_limits = [-2.0 + safe_distance, 5.0 - safe_distance]
        pos_y_limits = [-3.0 + safe_distance, 3.0 - safe_distance]
        pos_z_limits = [0.05, 3.0]  # Z limit slightly above ground

        stop_policy = False
        reason = ""

        # Check position limits
        if not (pos_x_limits[0] < position[0] < pos_x_limits[1]):
            stop_policy = True
            reason = f"Position X limit exceeded: {position[0]:.2f} outside [{pos_x_limits[0]}, {pos_x_limits[1]}]"
        elif not (pos_y_limits[0] < position[1] < pos_y_limits[1]):
            stop_policy = True
            reason = f"Position Y limit exceeded: {position[1]:.2f} outside [{pos_y_limits[0]}, {pos_y_limits[1]}]"
        elif not (pos_z_limits[0] < position[2] < pos_z_limits[1]):
            stop_policy = True
            reason = f"Position Z limit exceeded: {position[2]:.2f} outside [{pos_z_limits[0]}, {pos_z_limits[1]}]"

        if stop_policy:
            rospy.logwarn(
                f"Policy stopped due to state limit violation: {reason}"
            )
            self.start_policy = False
            # Optionally, publish a message indicating policy stopped
            # self.policy_status_pub_.publish(Bool(False)) # Need to add this publisher

    def save_data(
        self,
    ):
        if self.save_dir is not None:
            print("Saving rollout...")
            self.log_df = pd.DataFrame(
                self.log_list, columns=self.log_df.columns
            )
            os.makedirs(self.save_dir, exist_ok=True)
            datestr = f"{datetime.datetime.now():%Y%m%d-%H%M%S}"
            iter_num = rospy.get_param("~iter_num")
            iterstr = f"{iter_num}_"

            if self.use_sim_time:
                filename = "sim_" + iterstr + datestr + ".csv"
            else:
                filename = "real_" + iterstr + datestr + ".csv"

            self.log_df.to_csv(self.save_dir + filename, index=False)
            print("Data saved")

    @staticmethod
    def step_func(x):
        return 2.0 / (1 + np.exp(5 * x))

    @staticmethod
    def decay_func(x):
        return 2.0 * np.exp(-x)


def visualize_env(env, policy_dir, iter_num):
    obs = env.getObs()
    obs_norm = mask_out_lidar_scan(obs, nr_lidar_bins=32, nr_bins_to_mask=22)
    viz_quad_pos_xy = -np.ones((20, 2))
    quadstate = np.zeros([env.num_envs, 42], dtype=np.float64)
    env.wrapper.getQuadState(quadstate)
    # Fill in masked lidar bins
    obs_norm_padded = np.zeros(
        (obs_norm.shape[0], obs_norm.shape[1] + 22), dtype=np.float32
    )
    obs_norm_padded[:, : (-32 + 5)] = obs_norm[:, :-5]
    obs_norm_padded[:, -5:] = obs_norm[:, -5:]
    unnormalized_obs = env.unnormalize_obs(obs_norm_padded)
    unnormalized_obs *= obs_norm_padded != 0

    tree_poses = env.wrapper.getObstaclePoses()
    img_path = policy_dir / f"visualization/initial_step{iter_num}.png"
    print(f"Saving image to {img_path}")
    visualize_step(
        img_path,
        images=None,
        unnormalized_obs=unnormalized_obs,
        event_reprs=None,
        tree_poses=tree_poses,
        quadstate=quadstate,
        env_id=0,
        done=None,
        info=None,
        viz_quad_pos_xy=viz_quad_pos_xy,
        moving_top=False,
        plot_limits=[-6.5, 13.5, -6.5, 8.0],
    )


def main():
    # Debug
    # import debugpy
    # debugpy.listen(('0.0.0.0', 5678))
    # print("Waiting for debugger attach")
    # debugpy.wait_for_client()
    # random_seed = 0
    # policy_name = "PPO_6_CommandLooking_MaskLiDAR_Nature_Raven"
    # epoch_num = 8000

    rospy.init_node("flightpilot", anonymous=True)
    torch.set_num_threads(1)

    trial_num = rospy.get_param("~trial_num")
    iter_num = rospy.get_param("~iter_num")
    random_seed = rospy.get_param("~random_seed")

    # set random seed
    np.random.seed(random_seed)

    # set up directories
    # data_dir = os.environ["DATA_DIR"]
    # policy_dir = (Path(data_dir) / "policies" / "deployment" / policy_name)
    flightmare_path = os.environ["FLIGHTMARE_PATH"]
    data_dir = os.path.join(
        flightmare_path, "flightpilot", "real_world_policies"
    )
    policy_dir = Path(data_dir) / f"PPO_{trial_num}"
    quad_cfg_path = policy_dir / "quad_config.yaml"
    # render_cfg_path = policy_dir / "config_render_obstacle_avoidance.yaml"
    weights_path = policy_dir / "Policy" / "iter_{0:05d}.pth".format(iter_num)
    rms_path = policy_dir / "RMS" / ("iter_{0:05d}.npz".format(iter_num))
    vision_cfg_path = policy_dir / "config_vision_teacher.yaml"

    # environment configurations
    # render_cfg = YAML().load(open(render_cfg_path, "r"))
    env_cfg = YAML().load(open(vision_cfg_path, "r"))
    quad_cfg = YAML().load(open(quad_cfg_path, "r"))

    env_cfg["main"]["num_envs"] = 1
    env_cfg["main"]["test_env"] = "yes"

    # create policy
    policy_template = MlpPolicy

    # --- with pytorch JIT compilation
    device = get_device()
    saved_variables = torch.load(weights_path, map_location=device)
    policy = policy_template(**saved_variables["data"])
    policy.action_net = torch.nn.Sequential(policy.action_net, torch.nn.Tanh())
    # policy.build()
    policy.load_state_dict(saved_variables["state_dict"], strict=True)
    policy.to(device)
    policy.eval()

    print("\nLOADED POLICY\n")

    # policy just in time compilation
    #  dummy_inputs = torch.rand(1, obs_dim, device=device)
    #  policy = torch.jit.trace(policy, dummy_inputs)

    # * --- ENVIRONMENT --------------------------------------------------------
    # Commanded velocity is max velocity
    env_cfg["task"]["velocity_cmd_min"] = 7.0
    env_cfg["task"]["velocity_cmd_max"] = 7.0

    env = wrapper.ObstacleEnvVec(
        ObstacleAvoidanceVecEnv_v0(dump(env_cfg, Dumper=RoundTripDumper))
    )

    # Set Obstacles
    # fixed_obstacle_poses = np.array(
    #     [
    #         [
    #             [-1, -1, 2.85, 0.0, 0.0, 0.0, 1.0, 0.5],
    #             [8, -2, 2.85, 0.0, 0.0, 0.0, 1.0, 0.5]
    #         ],
    #     ],
    #     dtype=np.float64,
    # )
    fixed_obstacle_poses = np.array(
        [
            [
                [2, 0, 1.85, 0.0, 0.0, 0.0, 1.0, 0.5],
            ],
        ],
        dtype=np.float64,
    )
    env.set_obstacle_poses(fixed_obstacle_poses)
    # env.wrapper.getObstaclePoses()

    # Set the quadrotor state
    quad_state = env.getQuadState()
    quad_state[0, :3] = np.array([-5.0, 4.7, 1.11])
    quad_state[0, 3:7] = np.array([0.9238795, 0, 0, -0.3826834])
    env.setQuadState(quad_state)

    # load running mean standard deviation
    env.load_rms(rms_path)
    env.seed(seed=env_cfg["main"]["seed"])

    print("\nLOADED ENVIRONMENT\n")

    # visualize_env(env, policy_dir)
    # * --- End ENVIRONMENT ----------------------------------------------------

    # racing ROS pilot
    ObstacleAvoidancePilot(
        policy_dir=policy_dir,
        policy=policy,
        env=env,
        obstacle_poses=fixed_obstacle_poses,
        env_config=env_cfg,
        quad_config=quad_cfg,
        device=device,
    )

    # -- ros spin
    rospy.spin()


if __name__ == "__main__":
    main()
