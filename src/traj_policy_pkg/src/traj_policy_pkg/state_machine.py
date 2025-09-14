


from collections import deque
from dataclasses import dataclass
from enum import Enum, auto
from queue import Queue, Empty
import threading
import time
from typing import Deque, Optional, Protocol, Tuple, Union

import numpy as np
from scipy.spatial.transform import Rotation as R

import environments.env_trajectory_tracking.trajectory_generation.quintics as quintics
from observation_models import ObservationData
import traj_policy_pkg.utilities as utilities
from traj_policy_pkg.types.drone import DroneState

G = 9.81
# --- States ---
class StateMachineState(Enum):
    STOPPED = auto()
    # ARMED = auto()
    RUNNING = auto()
    # RECOVERY = auto()

# --- Events ---
@dataclass
class Event: pass
@dataclass 
class RunRequestEvent(Event): pass
@dataclass
class StopRequestEvent(Event): pass
@dataclass
class DroneStateUpdateEvent(Event):
    t: float
    drone_state: DroneState
    
# --- Effects ---
class LoggingLevel(Enum):
    INFO = "INFO"
    WARN = "WARN"
    ERROR = "ERROR"

class Effects(Protocol):
    def go_to_origin(self): ...
    def reset_observation(self): ...
    def push_observation(self, observation_data: ObservationData): ...
    def run_policy(self) -> Tuple[np.ndarray, np.ndarray]: ...
    def logging(self, message: str, level: LoggingLevel = LoggingLevel.INFO): ...
    def get_ros_time(self) -> float: ...

# --- Event Loop ---
MAX_QUEUE_SIZE = 10
PROCESS_BURST = 5
class EventLoop:
    """ This class handles the event queue and processes events. """
    def __init__(self, sm: "StateMachine"):
        self.state_machine = sm
        self.event_queue: Queue[Event] = Queue(maxsize=MAX_QUEUE_SIZE)
        self._latest_telem: Optional[DroneStateUpdateEvent] = None
        self._has_telem: bool = False

    def push_event(self, event: Event):
        if isinstance(event, DroneStateUpdateEvent):
            self._latest_telem = event
            self._has_telem = True
            return
        try:
            self.event_queue.put_nowait(event)
        except: pass
        
    def process_event(self):
        """ Process events in the queue. This should be called in a loop. """
        processed = 0
        while processed < PROCESS_BURST:
            try:
                evt = self.event_queue.get_nowait()
            except Empty:
                break
            self.state_machine.on_event(evt)
            processed += 1
        
        if self._has_telem:
            assert self._latest_telem is not None
            evt = self._latest_telem
            self._has_telem = False
            self.state_machine.on_event(evt)

# --- Constants / Parameters ---
ALL_DRONE_POSITION_BOUNDS = (
    np.array([-2., -2., -0.5]),
    np.array([2, 2, 2])
)

TO_RUN_DRONE_POSITION_BOUNDS = (
    np.array([-0.5, -0.5, -0.5]),
    np.array([0.5, 0.5, 0.5])
)
TO_RUN_DRONE_VELOCITY_BOUNDS = (
    np.array([-0.1, -0.1, -0.1]),
    np.array([0.1, 0.1, 0.1])
)
TO_RUN_DRONE_BODY_RATE_BOUNDS = (
    np.array([-0.1, -0.1, -0.1]),
    np.array([0.1, 0.1, 0.1])
)

# --- Trigger Conditions ---
RUNNING_DRONE_POSITION_BOUNDS = (
    np.array([-1.5, -1.5, -0.4]),
    np.array([1.5, 1.5, 2])
)
RUNNING_DRONE_MAX_ANGLE = np.deg2rad(60)  # 45 degrees in radians 

# --- Trajectory Parameters ---
TRAJ_ALPHA = 1.0
TRAJ_T_MIN = 0.5
TRAJ_NODE_SAMPLE_BOUNDS = (np.array([-1, -1, -0.5]), np.array([1, 1, 1]))

# --- State Machine ---
class StateMachine:
    """
    Explanation of the state machine states:
    - STOPPED: The state machine is not running. It can transition to ARMED with an arm command.
    - RUNNING: The state machine is actively running the policy. It can transition to STOPPED with a stop command or if the drone state goes out of bounds.
    """
    def __init__(self, effects: Effects, sampling_frequency: float = 10.0, start_check_window_duration: float = 1.0):
        self._buf_lock = threading.Lock()
        # Parameters
        self.SAMPLING_FREQUENCY = sampling_frequency
        self.START_CHECK_WINDOW_DURATION = start_check_window_duration
        self.BUFFER_DRONE_HISTORY_SIZE = int(self.SAMPLING_FREQUENCY * self.START_CHECK_WINDOW_DURATION * 1.5)
        
        # State
        self.state = StateMachineState.STOPPED
        self.effects: Effects = effects

        # Buffers for drone states
        self.last_time: Optional[float] = None
        self.last_drone_state: Optional[DroneState] = None
        self.last_policy_request: Optional[np.ndarray] = np.zeros((4,))
        self.last_command_request: Optional[np.ndarray] = np.array([G, 0,0,0])
        self.drone_state_buffer: Deque[DroneState] = deque(maxlen=self.BUFFER_DRONE_HISTORY_SIZE)
        
        # Trajectory related
        self.current_traj_quintic: Optional[quintics.QuinticChain3D] = None
        self.traj_start_time: Optional[float] = None
        
        

    def on_event(self, event: Event):
        if isinstance(event, StopRequestEvent):
            self._handle_stop_request()
        elif isinstance(event, RunRequestEvent):
            self._handle_run_request()
        elif isinstance(event, DroneStateUpdateEvent):
            self._handle_policy_state_update(t=event.t, drone_state=event.drone_state)
        else:
            raise ValueError("Unknown event type")

    def _handle_run_request(self):
        if self.state in (StateMachineState.STOPPED,):
            if not self._validate_pre_run_conditions():
                return
            self.state = StateMachineState.RUNNING
            self.effects.reset_observation()
            self.effects.logging("SM running.")
        else:
            self.effects.logging("Run request ignored in current state: " + str(self.state))

    def _handle_stop_request(self):
        if self.state in (StateMachineState.RUNNING, ):
            self.state = StateMachineState.STOPPED
            self.effects.reset_observation()
            self.effects.logging("SM stopped.")
        else:
            self.effects.logging("Stop request ignored in current state: " + str(self.state))

    def _handle_policy_state_update(self, t: float, drone_state: DroneState):

        delta_state_received_time = (self.effects.get_ros_time() - drone_state.time)
        t_start = time.perf_counter()

        self.last_time = t
        self.last_drone_state = drone_state
        # --- Update drone state buffer ---
        with self._buf_lock:
            self.drone_state_buffer.appendleft(drone_state)
        
        t_push_obs = time.perf_counter()
        
        # --- Transition based on state ---            
        drone_outside_safety_bounds = not utilities.is_within_bounds(drone_state.position, ALL_DRONE_POSITION_BOUNDS)
        if drone_outside_safety_bounds and self.state != StateMachineState.STOPPED:
            self.state = StateMachineState.STOPPED
            self.effects.logging(f"Drone outside safety bounds ({drone_state.position}). Stopping policy.", LoggingLevel.WARN)
                
        elif self.state == StateMachineState.RUNNING:
            if not self._check_running_conditions():
                self.state = StateMachineState.STOPPED
                self.effects.logging("Running conditions not met. Stopping policy.", LoggingLevel.WARN)
                
        
        # --- Execute effects based on state ---
        t_state_transitions = time.perf_counter()
        t_inf_time = None
        if self.state == StateMachineState.STOPPED:
            self.current_traj_quintic = None
            self.traj_start_time = None
            
        elif self.state == StateMachineState.RUNNING:
            desired_pos, desired_vel = self._desired_from_segment(t, drone_state)
            
            # --- Push observation data ---
            observation_data = ObservationData(
                drone_state_noisy=drone_state,
                last_policy_request=self.last_policy_request,
                desired_position=desired_pos,
                desired_velocity=desired_vel
            )
            self.effects.push_observation(observation_data)
            
            # --- Run inference ---
            t_inf_start = time.perf_counter()
            self.last_policy_request, self.last_command_request = self.effects.run_policy()
            t_inf_time = time.perf_counter() - t_inf_start
        
        t_end = time.perf_counter()
        # logging_str = ("Time breakdown (ms): \n"
        #                   f"  Total: {(t_end - t_start)*1000:.6f} ms\n"
        #                   f"  Push Obs: {(t_push_obs - t_start)*1000:.6f} ms\n"
        #                   f"  State Transitions: {(t_state_transitions - t_push_obs)*1000:.6f} ms\n"
        #                   f"  Inference: {t_inf_time * 1000 if t_inf_time is not None else 0:.6f} ms\n"
        #                   f"  Effects: {(t_end - t_state_transitions)*1000:.6f} ms\n"
        #                   f"  Delta State Received Time: {delta_state_received_time*1000:.6f} ms\n"
        #                 )
        # self.effects.logging(logging_str, LoggingLevel.INFO)
        
        total = t_end - t_start
        PROCESSING_TIME_THRESHOLD = 0.02
        OBSERVATION_DELAY_THRESHOLD = 0.02
        if total > PROCESSING_TIME_THRESHOLD:
            logging_str = f"Total time exceeded threshold: {total*1000:.6f} ms"
            self.effects.logging(logging_str, LoggingLevel.WARN)

        if delta_state_received_time > OBSERVATION_DELAY_THRESHOLD:
            logging_str = f"Delta state received time exceeded threshold: {delta_state_received_time*1000:.6f} ms"
            self.effects.logging(logging_str, LoggingLevel.WARN)

    # --- Utils ---
    def _drone_angle_from_vertical(self, drone_state: DroneState) -> float:
        """
        Angle between the drone's body z-axis and the world vertical (z-axis).
        Returns radians in [0, pi].
        """
        qw, qx, qy, qz = drone_state.orientation_wxyz
        r = R.from_quat([qx, qy, qz, qw])  # normalizes internally

        # World-frame direction of the body z-axis
        ez_world = r.apply([0.0, 0.0, 1.0])

        # Angle to world vertical [0,0,1]
        c = np.clip(ez_world[2], -1.0, 1.0)  # = R[2,2]
        return float(np.arccos(c))  # in [0, pi]
    
    def _validate_pre_run_conditions(self) -> bool:
        # Perform necessary checks before starting the policy:
        # - There are enough samples and duration in the drone state buffer
        # - The drone is stationary (velocity and angular velocity are within a small threshold)
        # - The drone position is within the defined bounds
        
          # seconds
        if self.last_time is None:
            self.effects.logging("No last time available, cannot validate pre-arm conditions.")
            return False
        current_time = self.last_time
        MIN_SAMPLES = self.START_CHECK_WINDOW_DURATION * self.SAMPLING_FREQUENCY
        
        has_enough_duration = False
        num_samples_within_timeframe = 0
        failed_checks = False
        
        with self._buf_lock:
            for drone_state in self.drone_state_buffer:
                assert drone_state.time is not None, "Drone state in buffer has no time."
                assert drone_state.position is not None, "Drone state in buffer has no position."
                assert drone_state.velocity is not None, "Drone state in buffer has no velocity."
                assert drone_state.body_rate is not None, "Drone state in buffer has no body rate."

                if drone_state.time < current_time - self.START_CHECK_WINDOW_DURATION:
                    has_enough_duration = True
                    break
                num_samples_within_timeframe += 1
                
                if not utilities.is_within_bounds(drone_state.position, TO_RUN_DRONE_POSITION_BOUNDS):
                    self.effects.logging(f"Drone position out of bounds: {drone_state.position}", LoggingLevel.WARN)
                    failed_checks = True
                    break

                if not utilities.is_within_bounds(drone_state.velocity, TO_RUN_DRONE_VELOCITY_BOUNDS):
                    self.effects.logging(
                        f"Drone is not stationary (Linear Velocity = [{drone_state.velocity[0]}, {drone_state.velocity[1]}, {drone_state.velocity[2]}])",
                        LoggingLevel.WARN
                    )
                    failed_checks = True
                    break
                
                if not utilities.is_within_bounds(drone_state.body_rate, TO_RUN_DRONE_BODY_RATE_BOUNDS):
                    self.effects.logging(
                        f"Drone is not stationary (Angular Velocity = [{drone_state.body_rate[0]}, {drone_state.body_rate[1]}, {drone_state.body_rate[2]}])",
                        LoggingLevel.WARN
                    )
                    failed_checks = True
                    break
                
        if not has_enough_duration:
            self.effects.logging("Not enough duration in drone state buffer to arm policy.", LoggingLevel.WARN)
            return False
        
        if num_samples_within_timeframe < MIN_SAMPLES:
            self.effects.logging(f"Not enough samples in drone state buffer to arm policy. Currently {num_samples_within_timeframe}/{MIN_SAMPLES}", LoggingLevel.WARN)
            return False
        
        if failed_checks:
            self.effects.logging("Failed checks for arming policy.", LoggingLevel.WARN)
            return False
        
        return True
    
    def _check_running_conditions(self) -> bool:
        # Perform necessary checks while running the policy:
        # - The drone is within the defined bounds
        # - The drone angle is within the defined bounds
        assert self.last_drone_state is not None, "Last drone state is None while checking running conditions."
        
        if not utilities.is_within_bounds(self.last_drone_state.position, RUNNING_DRONE_POSITION_BOUNDS):
            self.effects.logging("Drone position out of bounds.", LoggingLevel.WARN)
            return False

        if self._drone_angle_from_vertical(self.last_drone_state) >= RUNNING_DRONE_MAX_ANGLE:
            self.effects.logging("Drone angle from vertical exceeds maximum allowed angle.", LoggingLevel.WARN)
            return False

        return True
    
    def _desired_from_segment(self, t: float, drone_state) -> Tuple[np.ndarray, np.ndarray]:
        """Ensures there is a valid current segment, advances it, and returns (pos, vel)."""
        RESAMPLE_AFTER_S = 1.0  # buffer after segment end

        traj = self.current_traj_quintic
        t0 = self.traj_start_time

        # If no active segment, make one starting now
        if traj is None or t0 is None:
            traj, t0 = self._new_segment(start_pos=drone_state.position, start_time=t)
            self.effects.logging("New trajectory segment generated.", LoggingLevel.INFO)
            self.current_traj_quintic, self.traj_start_time = traj, t0
            return traj.at_time(0.0)

        # We have an active segment — check progress/time
        total = float(traj.segment_duration_cum[-1])
        t_traj = t - t0

        # Past the segment? Option A: immediately resample a new one
        if t_traj >= total + RESAMPLE_AFTER_S:
            self.effects.logging("Trajectory segment completed. Resampling.", LoggingLevel.INFO)
            traj, t0 = self._new_segment(start_pos=drone_state.position, start_time=t)
            self.current_traj_quintic, self.traj_start_time = traj, t0
            return traj.at_time(0.0)

        # Still within (or slightly beyond) segment: clamp to valid domain for evaluation
        pos, vel = traj.at_time(t_traj)
        return pos, vel
    
    def _new_segment(self, start_pos, start_time: float):
        key = int(np.random.randint(0, 1_000_000))

        traj = quintics.generate_quintic_chain(
            key=key,
            K=1,
            pos_box=TRAJ_NODE_SAMPLE_BOUNDS,
            T_min=TRAJ_T_MIN,
            alpha=TRAJ_ALPHA,
            start_point=start_pos,  # zero vel/acc assumed inside generator
        )
        return traj, start_time