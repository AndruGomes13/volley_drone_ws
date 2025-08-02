import json
from pathlib import Path
from typing import Deque, Optional, Tuple
import threading
import numpy as np
from agiros_msgs.msg import QuadState, Command, PolicyState
from policy_package.action.policy_to_command import ActionModelConfig, PolicyToNormalizedThrustAndBodyRate
from policy_package.observation.ObservationConfig import ObservationConfig
from policy_package.observation.observation import DroneViconObs
from policy_package.observation.observation_data import ObservationData
from policy_package.observation.observation_history import make_history_cls
from policy_package.observation.observation_populator import populate_observation
from policy_package.inference_server.PolicyServerInterface import PolicyServerInterface
from policy_package.types.ball import BallState
from policy_package.types.drone import DroneState
from std_msgs.msg import Bool
import rospy
from collections import deque
import time
import policy_package.utilities as utils



class MLPPilot:
    def __init__(self, quad_name:str, policy_sampling_frequency: float, jax_policy_path: Path, drone_only: bool = True):
        
        self._buf_lock = threading.Lock()
        self.quad_name = quad_name
        self.drone_only = drone_only
        config_path = jax_policy_path / "run_config.json"
        checkpoint_path = self._get_checkpoint_path(jax_policy_path)
        self.action_config, self.observation_config = self.parse_config(config_path)

        # --- Some Parameters ---
        self.SAMPLING_FREQUENCY= policy_sampling_frequency
        self.START_CHECK_WINDOW_DURATION = 1.0 # Seconds
        
        self.position_bounds = (
            np.array([-1.0, -1.0, 0.5]),
            np.array([1.0, 1.0, 2.0])
        )
        
        # --- State Variables ---
        self.run_policy = False
        
        # --- Action Model --- 
        self.policy_to_command = PolicyToNormalizedThrustAndBodyRate(self.action_config)

        
        # --- Observation Model ---
        self.observation_model = DroneViconObs
        self.observation_history_class = make_history_cls( 
            self.observation_model,
            lengths=self.observation_config.history_length_actor,
            delay=0
        )
        self.observation_history = self.observation_history_class.generate_zero()
        self.OBSERVATION_SHAPE = self.observation_history.to_array().shape

        # --- State Process Buffers ---
        self.last_drone_state: Optional[DroneState] = None
        self.last_ball_state: Optional[BallState] = None
        self.last_command: Optional[Command] = None
        self.last_policy_request: Optional[np.ndarray] = np.zeros((4,))
        
        
        # --- Buffer Initialization ---
        # These buffers are separate from the observation history. These are mainly for pre-activation checks.
        self.BUFFER_DRONE_HISTORY_SIZE = int(self.SAMPLING_FREQUENCY * self.START_CHECK_WINDOW_DURATION * 1.5)

        self.drone_state_buffer: Deque[DroneState] = deque(maxlen=self.BUFFER_DRONE_HISTORY_SIZE)

        # --- Setup jax policy server ---
        self.inference_server = PolicyServerInterface(
            checkpoint_path=checkpoint_path,
            observation_shape=self.OBSERVATION_SHAPE)

        self._check_inference_time()
    
        # --- ROS Initialization ---
        self.init_subscriptions()
        self.init_publishers()
        self.init_timers()
        
        rospy.loginfo("MLP Pilot initialized with policy sampling frequency: %f Hz", self.SAMPLING_FREQUENCY)
    
    def init_subscriptions(self):
        if self.drone_only:
            self.drone_state_sub = rospy.Subscriber(self.quad_name + "/agiros_pilot/state", QuadState, self.callback_drone_state)
        else:
            self.policy_state_sub = rospy.Subscriber(self.quad_name + "/agiros_pilot/policy_state", PolicyState, self.callback_policy_state)
        
        self.start_signal_sub = rospy.Subscriber(
            self.quad_name + "/run_policy", Bool, self.callback_run_signal, queue_size=1
        )
        
    def init_publishers(self):
        self.command_pub = rospy.Publisher(self.quad_name + "/agiros_pilot/feedthrough_command", Command, queue_size=1, tcp_nodelay=True)
        self.status_pub = rospy.Publisher(self.quad_name + "/policy_status", Bool, queue_size=1, tcp_nodelay=True)

    def init_timers(self):
        rospy.Timer(rospy.Duration.from_sec(0.1), self.status_callback)
        
    # --- Callbacks ---
    def callback_drone_state(self, msg: QuadState):
        assert self.drone_only, "Drone state callback should only be used in drone-only mode."
        # Process the state message
        drone_state = DroneState.from_msg(msg)
        self.process_drone_state(drone_state)
        
        # Parse observation
        obs_data = self.parse_observation_data()
        obs = populate_observation(self.observation_model, obs_data)
        self.observation_history = self.observation_history.push(obs)
        
        # Get and publish command
        self.get_and_publish_command(self.observation_history.to_array())  
        
    def callback_policy_state(self, msg: PolicyState):
        assert not self.drone_only, "Policy state callback should only be used in full policy mode."
        
        # Parse the policy state message
        drone_state = DroneState.from_msg(msg.quad_state)
        ball_state = BallState.from_msg(msg.ball_state)
        
        # Process the states
        self.process_drone_state(drone_state)
        self.process_ball_state(ball_state)
        
        # Parse observation
        obs_data = self.parse_observation_data()
        obs = populate_observation(self.observation_model, obs_data)
        self.observation_history = self.observation_history.push(obs)
        
        # Get and publish command
        self.get_and_publish_command(self.observation_history.to_array())        
    
    def callback_run_signal(self, msg: Bool):
        # Start or stop the policy execution based on the received signal and start conditions
        candidate_start = msg.data
        
        if candidate_start and not self._validate_pre_start_conditions():
            rospy.logwarn("Pre-start conditions not met. Cannot start policy.")
            return    
        
        self.run_policy = msg.data
        if self.run_policy:
            rospy.loginfo("Start signal received, initiating policy execution.")
        else:
            rospy.loginfo("Stop signal received, halting policy execution.")
        
    def status_callback(self, event):
        """
        Publish the current status of the policy execution.
        """
        status_msg = Bool()
        status_msg.data = self.run_policy
        self.status_pub.publish(status_msg)
        
    # --- State Processing ---
    def process_drone_state(self, drone_state: DroneState):
        # Add the drone state to the history buffers
        with self._buf_lock:
            self.drone_state_buffer.appendleft(drone_state)
        
        # Check if the drone state is within the defined bounds
        self._check_drone_state(drone_state)
        
        self.last_drone_state = drone_state

    def process_ball_state(self, ball_state: BallState):        
        self.last_ball_state = ball_state
    
    # --- Observation Data ---
    def parse_observation_data(self)->ObservationData:
        """ Builds the struct that is used for populating the observation. """
        return ObservationData(
            drone_state=self.last_drone_state,
            ball_state=self.last_ball_state if not self.drone_only else None,
            last_policy_request=self.last_policy_request
        )

    
    # --- Inference ---
    def get_and_publish_command(self, obs:np.ndarray):
        if not self.run_policy:
            return
    
        jax_command = self.inference_server.run_inference(obs)  # Call the JAX policy with the observation
        self.last_policy_request = jax_command
        command = self.policy_to_command.map(jax_command, rospy.Time.now().to_sec())  # Convert the JAX command to normalized thrust and body rate
        self.command_pub.publish(command)


    # --- Utils ---
    def _check_drone_state(self, drone_state: DroneState):
        # Check if the state is within the acceptable range
        if not utils.is_within_bounds(drone_state.position, self.position_bounds):
            if self.run_policy:
                rospy.logwarn("Drone position out of bounds: %s", drone_state.position)
            self._stop_policy()
            
    def parse_config(self, config_path: Path) -> Tuple[ActionModelConfig, ObservationConfig]:
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
        
    def _stop_policy(self):
        if self.run_policy:
            rospy.loginfo("Policy execution stopped.")
        self.run_policy = False
        
    def _validate_pre_start_conditions(self) -> bool:
        # Perform necessary checks before starting the policy:
        # - There are enough samples and duration in the drone state buffer
        # - The drone is stationary (velocity and angular velocity are within a small threshold)
        # - The drone position is within the defined bounds
        
          # seconds
        MIN_SAMPLES = self.START_CHECK_WINDOW_DURATION * self.SAMPLING_FREQUENCY
        current_time = rospy.get_time()
        
        has_enough_duration = False
        num_samples_within_timeframe = 0
        failed_checks = False
        
        with self._buf_lock:
            for drone_state in self.drone_state_buffer:
                if drone_state.time < current_time - self.START_CHECK_WINDOW_DURATION:
                    has_enough_duration = True
                    break
                num_samples_within_timeframe += 1
                
                if not utils.is_within_bounds(drone_state.position, self.position_bounds):
                    rospy.logwarn("Drone position out of bounds: %s", drone_state.position)
                    failed_checks = True
                    break
                
                if not utils.is_within_bounds(drone_state.velocity, (np.array([-0.1, -0.1, -0.1]), np.array([0.1, 0.1, 0.1]))):
                    rospy.logwarn(
                        "Drone is not stationary (Linear Velocity = [%f, %f, %f])",
                        drone_state.velocity[0],
                        drone_state.velocity[1],
                        drone_state.velocity[2]
                    )
                    failed_checks = True
                    break
                
                if not utils.is_within_bounds(drone_state.body_rate, (np.array([-0.1, -0.1, -0.1]), np.array([0.1, 0.1, 0.1]))):
                    rospy.logwarn(
                        "Drone is not stationary (Angular Velocity = [%f, %f, %f])",
                        drone_state.body_rate[0],
                        drone_state.body_rate[1],
                        drone_state.body_rate[2]
                    )
                    failed_checks = True
                    break
                
        if not has_enough_duration:
            rospy.logwarn("Not enough duration in drone state buffer to start policy.")
            return False
        
        if num_samples_within_timeframe < MIN_SAMPLES:
            rospy.logwarn(f"Not enough samples in drone state buffer to start policy. Currently {num_samples_within_timeframe}/{MIN_SAMPLES}")
            return False
        
        if failed_checks:
            rospy.logwarn("Failed checks for starting policy.")
            return False
        
        return True

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
    
    def _get_checkpoint_path(self, jax_policy_path: Path) -> Path:
        """
        Expects there is only a single checkpoint directory in the jax_policy_path. No name convention is expected.
        """
        assert jax_policy_path.is_dir(), f"JAX policy path {jax_policy_path} is not a directory."
        checkpoint_dirs = [d for d in jax_policy_path.iterdir() if d.is_dir()]
        assert len(checkpoint_dirs) == 1, f"Expected exactly one checkpoint directory in {jax_policy_path}, found {len(checkpoint_dirs)}."
        return checkpoint_dirs[0]
