from enum import Enum, auto
import json
import math
from pathlib import Path
from typing import Deque, Optional, Tuple, Union
import threading
import numpy as np
import agiros_msgs.msg as msg
from traj_policy_pkg.state_machine import StateMachine, LoggingLevel, StateMachineState, EventLoop, DroneStateUpdateEvent, StopRequestEvent, RunRequestEvent
from traj_policy_pkg.inference_server.PolicyServerInterface import PolicyServerInterface
from traj_policy_pkg.types.drone import DroneState

# ---
from action_models import Command, ActionModelConfig
from action_models.policy_mapper import PolicyToNormalizedThrustAndBodyRate
from observation_models import ObservationConfig, ObservationData, make_history_cls, observation_class_from_type, ObservationHistory
# from sim_types.DroneState import DroneState

# ---


from std_msgs.msg import String, Bool, Empty
import rospy
import time
import geometry_msgs.msg as geometry_msgs

DEG= np.pi / 180  # Degrees to radians conversion
G = 9.81
class PolicyInterface:
    """
    An interface for the policy which configures the action and observation models, initializes the policy server, and manages the state of the policy and observation history.
    """

    def __init__(self, policy_path:Path, sampling_frequency: float):
        self.SAMPLING_FREQUENCY = sampling_frequency
        
        config_path = policy_path / "run_config.json"
        checkpoint_path = self._get_checkpoint_path(policy_path)
        self.action_config, observation_config = self._parse_config(config_path)
        
        rospy.loginfo(f"Using action model config: {self.action_config}")
        
        # --- Action Model ---
        self.policy_to_command = PolicyToNormalizedThrustAndBodyRate(self.action_config)
        self.clipping_values  = np.array(self.action_config.max_command_rate_change) * 1/self.SAMPLING_FREQUENCY

        # --- Observation Model ---
        self.observation_model = observation_class_from_type(observation_config.actor_observation_type)
        self.observation_history_class = make_history_cls( 
            self.observation_model,
            lengths=observation_config.history_length_actor,
            delay=0
        )
        self.observation_history: ObservationHistory = self.observation_history_class.generate_zero()
        self.OBSERVATION_SHAPE = self.observation_history.to_array().shape
        
        # --- Start Server ---
        self.inference_server = PolicyServerInterface(
            checkpoint_path=checkpoint_path,
            observation_shape=self.OBSERVATION_SHAPE
        )
        
        # --- Buffer ---
        self.last_command: Optional[Command] = None
        
        self._check_inference_time()
        
    def push_observation(self, observation_data: ObservationData):
        """
        Pushes the observation data to the observation history.
        """
        obs = self.observation_model.from_observation_data(observation_data)
        self.observation_history = self.observation_history.push(obs)

    def get_command(self) -> Tuple[Command, np.ndarray]:
        """
        Runs the inference on the observation and returns the body rate command and the policy command.
        """
        obs = self.observation_history.to_array()
        policy_request = self.inference_server.run_inference(obs)
        time = rospy.Time.now().to_sec()
        request_command = self.policy_to_command.map(policy_request, time)
        
        self.last_command = request_command
        
        return request_command, policy_request

    # --- Utility Methods ---
    def _get_checkpoint_path(self, policy_path: Path) -> Path:
        """
        Expects there is only a single checkpoint directory in the policy_path. No name convention is expected.
        """
        assert policy_path.is_dir(), f"Policy path {policy_path} is not a directory."
        checkpoint_dirs = [d for d in policy_path.iterdir() if d.is_dir()]
        assert len(checkpoint_dirs) == 1, f"Expected exactly one checkpoint directory in {policy_path}, found {len(checkpoint_dirs)}."
        return checkpoint_dirs[0]
    
    def _parse_config(self, config_path: Path) -> Tuple[ActionModelConfig, ObservationConfig]:
        """
        Parse the configuration file for action and observation models.
        """
        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        
        with open(config_path, 'r') as f:
            config_data = json.load(f)
        env_data = config_data.get("env", {})

        action_config = ActionModelConfig(**env_data.get("action_model_config"))
        observation_config = ObservationConfig(**env_data.get("observation_config"))
        
        return action_config, observation_config
        
    def _check_inference_time(self):
        ''' 
        Check the inference time of the JAX policy. Will account for:
        - Serialization and deserialization of the state
        - Server communication overhead
        - JAX policy inference time
        '''
        
        expected_inference_time = 1 / self.SAMPLING_FREQUENCY
        random_obs_hist = self.observation_history_class.generate_random(0)
        
        # --- Run inference ---
        NUM_ITER = 1000
        inference_times = []
        for _ in range(NUM_ITER):
            start = time.perf_counter()
            self.inference_server.run_inference(random_obs_hist.to_array())

            end = time.perf_counter()
            inference_times.append(end - start)
        inference_times = np.array(inference_times)
        
        # --- Calculate and log statistics ---
        # --- Mean 
        avg_inference_time = np.sum(inference_times) / NUM_ITER
        print(f"JAX policy inference time: {avg_inference_time * 1000:.6f} ms per iteration")
        
        # --- Percentiles
        p = [50, 75, 90, 95, 99]
        percentiles = np.percentile(inference_times, p)
        for i, perc in enumerate(p):
            print(f"JAX policy inference time {perc}th percentile: {percentiles[i] * 1000:.6f} ms")

        if avg_inference_time < expected_inference_time * 0.5:
            rospy.loginfo(f"JAX policy inference time is acceptable: {avg_inference_time * 1000:.6f} ms per iteration")
        elif avg_inference_time < expected_inference_time * 0.8:
            rospy.logwarn(f"JAX policy inference time is high: {avg_inference_time * 1000:.6f} ms per iteration")
        else:
            rospy.logerr(f"JAX policy inference time is too high: {avg_inference_time * 1000:.6f} ms per iteration")
            # raise RuntimeError(
            #     f"JAX policy inference time exceeds sampling period: {inference_time * 1000:.6f} ms per iteration"
            # )
        
        return avg_inference_time

class Effects:
    """ Effects class that implements the effects of the state machine."""
    def __init__(self, pilot: "MLPPilot") -> None:
        self.p = pilot

    def run_policy(self) -> Tuple[np.ndarray, np.ndarray]:
        return self.p._effect_run_policy()
    def go_to_origin(self):
        return self.p._effect_go_to_origin()
    def push_observation(self, observation_data: ObservationData):
        return self.p._effect_push_observation(observation_data)
    def reset_observation(self):
        return self.p._effect_reset_observation()
    def logging(self, message:str, level:LoggingLevel = LoggingLevel.INFO):
        self.p._effect_logging(message, level)
    def get_ros_time(self) -> float:
        return self.p._get_ros_time()
    
ORIGIN_OFFSET = np.array([0.5, 0.0, 1.2])  # Offset to origin in the world frame

class MLPPilot:
    """
    MLP Pilot class that manages the policy execution, state management, and communication with the ROS system.
    It uses the PolicyInterface to interact with the JAX policy server and manages the drone states.
    """  
    def __init__(self, quad_name: str, policy_sampling_frequency: float, policy_path: Path, traj_policy_name: Path):
        self.quad_name = quad_name
        
        # --- Policy Paths ---
        traj_policy_name = policy_path / traj_policy_name
        self.traj_policy = PolicyInterface(traj_policy_name, policy_sampling_frequency)
        
        # --- Some Parameters ---
        self.SAMPLING_FREQUENCY = policy_sampling_frequency
        self.START_CHECK_WINDOW_DURATION = 1.0
        
        
        # --- State Variables ---  
        effects = Effects(self)          
        self.state_machine = StateMachine(effects=effects, sampling_frequency=self.SAMPLING_FREQUENCY, start_check_window_duration=self.START_CHECK_WINDOW_DURATION)
        self.event_loop = EventLoop(self.state_machine)
        
        # --- Start Pub/Sub ---
        self.init_subscriptions()
        self.init_publishers()
        self.init_timers()
        
    def init_subscriptions(self):
        self.policy_state_sub = rospy.Subscriber(self.quad_name + "/agiros_pilot/policy_state", msg.PolicyState, self.callback_policy_state, queue_size=1, tcp_nodelay=True)
        
        #NOTE: Ideally we want arm and stop to be sent by the agiros gui separately. At the moment, to avoid changing the gui, we will use the same convention where the arm/start and stop signal are sent on the same topic as a bool message.
        # self.arm_signal_sub = rospy.Subscriber(
        #     self.quad_name + "/arm_policy", Empty, self.callback_arm_signal, queue_size=1
        # )
        
        # self.stop_signal_sub = rospy.Subscriber(
        #     self.quad_name + "/stop_policy", Empty, self.callback_stop_signal, queue_size=1
        # )
        
        self.start_signal_sub = rospy.Subscriber(
            self.quad_name + "/run_policy", Bool, self.callback_run_signal, queue_size=1
        )
        
    def init_publishers(self):
        self.command_pub = rospy.Publisher(self.quad_name + "/agiros_pilot/feedthrough_command", msg.Command, queue_size=1, tcp_nodelay=True)
        self.policy_sm_state_pub = rospy.Publisher(self.quad_name + "/policy_sm_state", String, queue_size=1, tcp_nodelay=True)
        self.go_to_pose_pub = rospy.Publisher(self.quad_name + "/agiros_pilot/go_to_pose", geometry_msgs.PoseStamped, queue_size=1, tcp_nodelay=True)

    def init_timers(self):
        rospy.Timer(rospy.Duration.from_sec(0.1), self.status_callback)
    
    def tick(self):
        self.event_loop.process_event()
        
    # --- Callbacks ---
    def callback_policy_state(self, msg: msg.PolicyState):
        # Parse the policy state message
        t = rospy.Time.now().to_sec()
        drone_state = DroneState.from_msg(msg.quad_state)
        
        # Subtract offset
        drone_state = drone_state.replace(position=drone_state.position - ORIGIN_OFFSET)
        # drone_state.position = drone_state.position - np.array(ORIGIN_OFFSET)
        
        # Parse event
        state_event = DroneStateUpdateEvent(
            t=t,
            drone_state=drone_state,
        )
        
        self.event_loop.push_event(state_event)
        
    def status_callback(self, event):
        """
        Publish the current status of the policy execution.
        """
        status_msg = String()
        if self.state_machine.state == StateMachineState.STOPPED:
            status_msg.data = "STOPPED"
        elif self.state_machine.state == StateMachineState.RUNNING:
            status_msg.data = "RUNNING"
            
        self.policy_sm_state_pub.publish(status_msg)
    
    def callback_run_signal(self, msg: Bool):
        """
        Callback for the run signal. Starts the policy if the message is True and stops it if False. NOTE: This is a temporary solution to avoid changing the GUI.
        """
        if msg.data:
            self.callback_sun_signal(Empty())  # Simulate arm signal if run signal is True
        else:
            self.callback_stop_signal(Empty())
    
    def callback_sun_signal(self, msg: Empty):
        arm_event = RunRequestEvent()
        self.event_loop.push_event(arm_event)
    
    def callback_stop_signal(self, msg: Empty):
        stop_event = StopRequestEvent()
        self.event_loop.push_event(stop_event)
        
    # --- Effects ---   
    def _effect_run_policy(self) ->Tuple[np.ndarray, np.ndarray]:
        command, last_policy_request = self.traj_policy.get_command()
        assert command.collective_thrust is not None, "Command collective thrust is None"
        assert command.body_rate is not None, "Command body rate is None"
        
        cmd_msg = msg.Command()
        cmd_msg.collective_thrust = command.collective_thrust
        cmd_msg.bodyrates.x = command.body_rate[0]
        cmd_msg.bodyrates.y = command.body_rate[1]
        cmd_msg.bodyrates.z = command.body_rate[2]
        self.command_pub.publish(cmd_msg)
        last_command_request = np.concatenate([command.collective_thrust, command.body_rate], axis=0)
        return last_policy_request, last_command_request
        
    def _effect_go_to_origin(self):
        """
        Sends a command to the drone to go to the origin position.
        """
        go_to_pose_msg = geometry_msgs.PoseStamped()
        go_to_pose_msg.pose.position.x = float(ORIGIN_OFFSET[0])
        go_to_pose_msg.pose.position.y = float(ORIGIN_OFFSET[1])
        go_to_pose_msg.pose.position.z = float(ORIGIN_OFFSET[2])
        heading = float(0)
        go_to_pose_msg.pose.orientation.w = math.cos(heading / 2.0)
        go_to_pose_msg.pose.orientation.z = math.sin(heading / 2.0)

        self.go_to_pose_pub.publish(go_to_pose_msg)
        rospy.loginfo("Command sent to go to origin position: %s", go_to_pose_msg.pose.position)
        
    def _effect_push_observation(self, observation_data: ObservationData):
        """
        Pushes the observation data to the policy interfaces.
        """
        self.traj_policy.push_observation(observation_data)
    
    def _effect_reset_observation(self):
        """
        Resets the observation history in the policy interfaces.
        """
        self.traj_policy.observation_history = self.traj_policy.observation_history_class.generate_zero() 
    
    def _effect_logging(self, message: str, level: LoggingLevel = LoggingLevel.INFO):
        """
        Log a message with the specified logging level.
        """
        if level == LoggingLevel.INFO:
            rospy.loginfo(message)
        elif level == LoggingLevel.WARN:
            rospy.logwarn(message)
        elif level == LoggingLevel.ERROR:
            rospy.logerr(message)
        else:
            rospy.logdebug(message)
        
    def _get_ros_time(self) -> float:
        """
        Returns the current ROS time.
        """
        return rospy.Time.now().to_sec()