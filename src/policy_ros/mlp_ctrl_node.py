from dataclasses import dataclass
import json
import os
from pathlib import Path
import subprocess
from typing import Deque, List, Tuple

import numpy as np
from agiros_msgs.msg import QuadState, Telemetry, Command
from observation import FullObservationWithBall
from state_interface import DroneState, BallState, StateHistory
from std_msgs.msg import Bool
import rospy
from collections import deque
import time
from protocol.messages import InferenceMsg, InferenceReply, parse_message
from select import select
    
class PolicyServerInterface:
    def __init__(self, jax_policy_path: Path, observation_shape: Tuple[int, ...]):
        self.jax_policy_path = jax_policy_path
        self.observation_shape = observation_shape
        
        self.inference_server = self._get_inference_server()
        
        self.timeout = 2000 / 1000  # seconds

    def _get_inference_server(self):
        # Start the inference server with the specified checkpoint directory and observation shape
        python_file = os.path.join(os.path.dirname(__file__), "inference_server.py")
        server = subprocess.Popen(
            ["python3.11", python_file, "--checkpoint-dir", self.jax_policy_path.absolute(), "--observation-shape", f"{str(self.observation_shape)}"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,  # line buffered
        )

        print("Subprocess Id:", server.pid)
        rospy.loginfo(f"Inference server started with checkpoint directory: {self.jax_policy_path}")
       
        if not server.stdout or not server.stderr:
            raise RuntimeError("Failed to start inference server: No stdout or stderr pipe")
        
        return server

    def run_inference(self, observation: np.ndarray) -> np.ndarray:
        if not self.inference_server:
            raise RuntimeError("Inference server is not running")
        
        if self.inference_server.poll() is not None:
            raise RuntimeError("Inference server has exited")
            
        try:
            inference_msg = InferenceMsg(obs=observation.tolist())
            self.send(inference_msg.to_json())
            
        except BrokenPipeError:
            raise RuntimeError("Inference server pipe is broken")
        
        ready, _, _ = select([self.inference_server.stdout], [], [], self.timeout)
        if not ready:
            raise TimeoutError("Inference server timed out")
        
        response = self.inference_server.stdout.readline()
        if response == "":                                # <- child closed pipe
            rc = self.inference_server.poll()
            raise RuntimeError(f"Inference server exited with code {rc}")

        response_data = InferenceReply.from_json(response)

        if isinstance(response_data, InferenceReply):
            if response_data.status == "ok":
                return np.array(response_data.result)
            else:
                raise RuntimeError(f"Inference server returned error: {response_data.error}")
        else:
            raise ValueError(f"Unexpected response type: {type(response_data)}")

    def send(self, message: dict):
        """Send a message to the inference server."""
        if not self.inference_server:
            raise RuntimeError("Inference server is not running")

        try:
            self.inference_server.stdin.write(json.dumps(message) + "\n")
            self.inference_server.stdin.flush()
        except BrokenPipeError:
            raise RuntimeError("Inference server pipe is broken")


class MLPPilot:
    def __init__(self, policy_sampling_frequency: float, jax_policy_path: Path):
        
        # --- Some Parameters ---
        self.SAMPLING_FREQUENCY= policy_sampling_frequency
        self.START_CHECK_WINDOW_DURATION = 1.0 # Seconds
        
        self.position_bounds = (
            np.array([-10.0, -10.0, -10.0]),
            np.array([10.0, 10.0, 10.0])
        )
        
        # --- State Variables ---
        self.run_policy = False
        
        # --- Action Model --- #TODO: Defined the actual scalings
        DEG = np.pi / 180.0
        MASS = ...
        t = ...
        wx = 600 * DEG
        wy = wx
        wz = 250 * DEG
        
        # --- Observation Model ---
        self.DRONE_HISTORY_LEN = 10
        self.BALL_HISTORY_LEN = 10
        self.ACTION_HISTORY_LEN = 10
        observation_model = FullObservationWithBall
        observation_model.resolve_fields(self.DRONE_HISTORY_LEN, self.BALL_HISTORY_LEN, self.ACTION_HISTORY_LEN)
        self.OBSERVATION_SHAPE = observation_model.get_observation_shape()
        
        # --- Buffer Initialization ---
        self.BUFFER_DRONE_HISTORY_SIZE = int(max(
            int(self.SAMPLING_FREQUENCY * self.START_CHECK_WINDOW_DURATION),
            self.DRONE_HISTORY_LEN
        ) * 1.2)
        self.BUFFER_BALL_HISTORY_SIZE = self.BALL_HISTORY_LEN
        self.BUFFER_ACTION_HISTORY_LEN = self.ACTION_HISTORY_LEN

        self.drone_state_history: Deque[DroneState] = deque(maxlen=self.BUFFER_DRONE_HISTORY_SIZE)
        self.ball_state_history: Deque[BallState] = deque(maxlen=self.BUFFER_BALL_HISTORY_SIZE)
        self.action_history: Deque[np.ndarray] = deque(maxlen=self.BUFFER_ACTION_HISTORY_LEN)
        
        # --- Setup jax policy server ---
        self.inference_server = PolicyServerInterface(
            jax_policy_path=jax_policy_path,
            observation_shape=self.OBSERVATION_SHAPE)

        self._check_inference_time()
    
        # --- ROS Initialization ---
        self.init_subscriptions()
        self.init_publishers()
    
    def init_subscriptions(self):
        self.state_sub = rospy.Subscriber("agiros_pilot/state", QuadState, self.callback_drone_state)
        self.telemetery_sub = rospy.Subscriber(
            "agiros_pilot/telemetry",
            Telemetry,
            queue_size=1,
        )
        self.start_signal_sub = rospy.Subscriber(
            "/start_policy", Bool, self.callback_start_signal, queue_size=1
        )
        
    def init_publishers(self):
        self.command_pub = rospy.Publisher("agiros_pilot/feedthrough_command", Command, queue_size=1, tcp_nodelay=True)
        
    def callback_drone_state(self, msg: QuadState):
        # Process the state message
        drone_state = self._parse_state_msg(msg)
        
        self.drone_state_history.appendleft(drone_state)
        
        # Check if the drone state is within the defined bounds
        self._check_drone_state(drone_state)
        
            
    def callback_telemetry(self, msg: Telemetry):
        if msg is not None:
            self.voltage = msg.voltage
        
    def callback_start_signal(self, msg: Bool):
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
       
    def _check_drone_state(self, drone_state: DroneState):
        # Check if the state is within the acceptable range
        if not self._is_within_bounds(drone_state.position, self.position_bounds):
            rospy.logwarn("Drone position out of bounds: %s", drone_state.position)
            self.stop_policy()

    def _is_within_bounds(self, value: np.ndarray, bounds: Tuple[np.ndarray, np.ndarray]) -> bool:
        return np.all(value >= bounds[0]) and np.all(value <= bounds[1])

    def _parse_state_msg(self, msg: QuadState) -> DroneState:
        # Parse the state message to extract relevant information
        time = msg.header.stamp.to_sec()
        position = np.array([msg.pose.position.x, msg.pose.position.y, msg.pose.position.z])
        velocity = np.array([msg.velocity.linear.x, msg.velocity.linear.y, msg.velocity.linear.z])
        orientation_wxyz = np.array([msg.pose.orientation.w, msg.pose.orientation.x, msg.pose.orientation.y, msg.pose.orientation.z])
        angular_velocity = np.array([msg.velocity.angular.x, msg.velocity.angular.y, msg.velocity.angular.z])
        angular_acceleration = np.array([msg.acceleration.angular.x, msg.acceleration.angular.y, msg.acceleration.angular.z])
        gyro_bias = np.array([msg.gyr_bias.x, msg.gyr_bias.y, msg.gyr_bias.z])
        

        return DroneState(
            time=time,
            position=position,
            velocity=velocity,
            orientation=orientation_wxyz,
            angular_velocity=angular_velocity,
            angular_acceleration=angular_acceleration,
            gyro_bias=gyro_bias
        )

    def _parse_ball_msg(self, msg) -> BallState:
        # TODO: Implement the parsing of the ball state message
        pass
    
    def get_and_publish_command(self):
        if not self.run_policy:
            return
    
            
        observation = ...
        jax_command = ...  # Call the JAX policy with the observation
        
        command = Command()
        command.bodyrates = jax_command.bodyrates.tolist()
        command.collective_thrust = jax_command.thrust.tolist()
        
        self.command_pub.publish(command)

    def _stop_policy(self):
        self.run_policy = False
        rospy.loginfo("Policy execution stopped.")
        
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
        
        for drone_state in self.drone_state_buffer:
            if drone_state.time < current_time - self.START_CHECK_WINDOW_DURATION:
                has_enough_duration = True
                break
            num_samples_within_timeframe += 1
            
            if not self._is_within_bounds(drone_state.position, self.position_bounds):
                rospy.logwarn("Drone position out of bounds: %s", drone_state.position)
                failed_checks = True
                break
            
            if not self._is_within_bounds(drone_state.velocity, (np.array([-0.1, -0.1, -0.1]), np.array([0.1, 0.1, 0.1]))):
                rospy.logwarn(
                    "Drone is not stationary (Linear Velocity = [%f, %f, %f])",
                    drone_state.velocity[0],
                    drone_state.velocity[1],
                    drone_state.velocity[2]
                )
                failed_checks = True
                break
            
            if not self._is_within_bounds(drone_state.angular_velocity, (np.array([-0.1, -0.1, -0.1]), np.array([0.1, 0.1, 0.1]))):
                rospy.logwarn(
                    "Drone is not stationary (Angular Velocity = [%f, %f, %f])",
                    drone_state.angular_velocity[0],
                    drone_state.angular_velocity[1],
                    drone_state.angular_velocity[2]
                )
                failed_checks = True
                break
            
        if not has_enough_duration:
            rospy.logwarn("Not enough duration in drone state buffer to start policy.")
            return False
        
        if num_samples_within_timeframe < MIN_SAMPLES:
            rospy.logwarn("Not enough samples in drone state buffer to start policy.")
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
        
        drone_state_history = deque(maxlen=self.BUFFER_DRONE_HISTORY_SIZE)
        ball_state_history = deque(maxlen=self.BUFFER_BALL_HISTORY_SIZE)
        action_history = deque(maxlen=self.BUFFER_ACTION_HISTORY_LEN)
        
        def _random_drone_state() -> DroneState:
            return DroneState(
                time=np.random.uniform(0, 10),
                position=np.random.uniform(-10, 10, size=3),
                velocity=np.random.uniform(-1, 1, size=3),
                orientation_wxyz=np.random.uniform(-1, 1, size=4),
                body_rate=np.random.uniform(-1, 1, size=3),
                angular_acceleration=np.random.uniform(-1, 1, size=3),
                gyro_bias=np.random.uniform(-0.1, 0.1, size=3)
            )
        def _random_ball_state() -> BallState:
            return BallState(
                time=np.random.uniform(0, 10),
                position=np.random.uniform(-10, 10, size=3),
                velocity=np.random.uniform(-1, 1, size=3)
            )
        # Fill the buffers with random states
        for _ in range(self.BUFFER_DRONE_HISTORY_SIZE):
            drone_state_history.append(_random_drone_state())
        for _ in range(self.BUFFER_BALL_HISTORY_SIZE):
            ball_state_history.append(_random_ball_state())
        for _ in range(self.BUFFER_ACTION_HISTORY_LEN):
            action_history.append(np.random.uniform(-1, 1, size=4))
        
        state_history = StateHistory(
            drone_state_history=drone_state_history,
            ball_state_history=ball_state_history,
            action_history=action_history)
        
        NUM_ITER = 100
        start_time = time.time()
        for _ in range(NUM_ITER):
            obs = FullObservationWithBall.get_observation(state_history)
            self.inference_server.run_inference(obs.to_array())
            
        end_time = time.time()
        inference_time = (end_time - start_time) / NUM_ITER

        if inference_time < expected_inference_time * 0.5:
            rospy.loginfo(f"JAX policy inference time is acceptable: {inference_time * 1000:.6f} ms per iteration")
        elif inference_time < expected_inference_time * 0.8:
            rospy.logwarn(f"JAX policy inference time is high: {inference_time * 1000:.6f} ms per iteration")
        else:
            rospy.logerr(f"JAX policy inference time is too high: {inference_time * 1000:.6f} ms per iteration")
            # raise RuntimeError(
            #     f"JAX policy inference time exceeds sampling period: {inference_time * 1000:.6f} ms per iteration"
            # )
        
        return inference_time
    
def main():
    CONTROL_RATE = 100  # Hz
    
    rospy.init_node("mlp_ctrl_node", anonymous=True)

    pilot = MLPPilot(CONTROL_RATE)
    rospy.loginfo("MLP Control Node initialized and running.")
    
    rate = rospy.Rate(CONTROL_RATE)  # 100 Hz

    while not rospy.is_shutdown():
        pilot.get_and_publish_command()
        rate.sleep()
        rospy.spin_once()
        
        
if __name__ == "__main__":
    # Run to test the inference server
    mpl_pilot = MLPPilot(policy_sampling_frequency=100, jax_policy_path=Path("/home/agilicious/catkin_ws/src/policy_ros/jiaxu_code.py"))
    
    while True:
        t = time.time()
        inf_time = mpl_pilot._check_inference_time()
        print(f"Inference result: {inf_time * 1000:.2f} ms")
        print("Inference executed successfully.")
        time.sleep(1)  # Adjust the sleep time as needed
    