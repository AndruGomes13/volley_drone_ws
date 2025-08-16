


from collections import deque
from dataclasses import dataclass
from enum import Enum, auto
from queue import Queue, Empty
import threading
from typing import Deque, Optional, Protocol

import numpy as np
from scipy.spatial.transform import Rotation as R

from bounce_policy_pkg.observation.observation_data import ObservationData
import bounce_policy_pkg.utilities as utilities
from bounce_policy_pkg.types.drone import DroneState
from bounce_policy_pkg.types.ball import BallState

# --- States ---
class StateMachineState(Enum):
    STOPPED = auto()
    ARMED = auto()
    RUNNING = auto()
    RECOVERY = auto()

# --- Events ---
@dataclass
class Event: pass
@dataclass 
class ArmRequestEvent(Event): pass
@dataclass
class StopRequestEvent(Event): pass
@dataclass
class PolicyStateUpdateEvent(Event):
    t: float
    drone_state: DroneState
    ball_state: BallState
    
# --- Effects ---
class LoggingLevel(Enum):
    INFO = "INFO"
    WARN = "WARN"
    ERROR = "ERROR"

class Effects(Protocol):
    def go_to_origin(self): ...
    def reset_observation(self): ...
    def push_observation(self, observation_data: ObservationData): ...
    def run_recovery_control(self) -> Optional[np.ndarray]: ...
    def run_bounce_control(self) -> np.ndarray: ...
    def logging(self, message: str, level: LoggingLevel = LoggingLevel.INFO): ...

# --- Event Loop ---
MAX_QUEUE_SIZE = 10
PROCESS_BURST = 5
class EventLoop:
    """ This class handles the event queue and processes events. """
    def __init__(self, sm: "StateMachine"):
        self.state_machine = sm
        self.event_queue: Queue[Event] = Queue(maxsize=MAX_QUEUE_SIZE)
        self._latest_telem: Optional[PolicyStateUpdateEvent] = None
        self._has_telem: bool = False

    def push_event(self, event: Event):
        if isinstance(event, PolicyStateUpdateEvent):
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

# --- Drone and Ball Bounds ---
# --- Constants / Parameters ---
# ALL_DRONE_POSITION_BOUNDS = (
#     np.array([-1.2, -1.2, -0.8]),
#     np.array([1.2, 1.2, 1])
# )
ALL_DRONE_POSITION_BOUNDS = (
    np.array([-2., -2., -0.5]),
    np.array([2, 2, 2])
)

TO_ARM_DRONE_POSITION_BOUNDS = (
    np.array([-0.5, -0.5, -0.5]),
    np.array([0.5, 0.5, 0.5])
)
TO_ARM_DRONE_VELOCITY_BOUNDS = (
    np.array([-0.1, -0.1, -0.1]),
    np.array([0.1, 0.1, 0.1])
)
TO_ARM_DRONE_BODY_RATE_BOUNDS = (
    np.array([-0.1, -0.1, -0.1]),
    np.array([0.1, 0.1, 0.1])
)

# --- Trigger Conditions ---
# Area in which the ball must be detected to trigger the start of the bouncing policy
ARMED_BALL_POSITION_TO_RUNNING_BOUNDS = (
    np.array([-1.4, -1.4, -0.6]),
    np.array([1.4, 1.4, 4])
)

# Area in which the ball must be detected to continue running the bouncing policy
RUNNING_BALL_POSITION_BOUNDS = (
 np.array([-1.5, -1.5, -1]),
    np.array([1.5, 1.5, 4])
)

RUNNING_DRONE_POSITION_BOUNDS = (
    np.array([-1, -1, -0.4]),
    np.array([1, 1, 1.5])
)
RUNNING_DRONE_MAX_ANGLE = np.deg2rad(60)  # 45 degrees in radians 

RECOVERY_DRONE_MAX_ANGLE = np.deg2rad(90)  # 60 degrees in radians

# --- State Machine ---
class StateMachine:
    """
    Explanation of the state machine states:
    - STOPPED: The state machine is not running. It can transition to ARMED with an arm command.
    - ARMED: The state machine is armed and ready to run. It can transition to RUNNING when the ball state is valid and within bounds.
    - RUNNING: The state machine is actively running the policy. It can transition to STOPPED with a stop command or if the drone or ball state goes out of bounds. I can also transition to RECOVERY if the drone or ball state goes out of bounds.
    - RECOVERY: The state machine is in recovery mode, typically after a failure or out-of-bounds condition. It can transition back ARMED or STOPPED based on the user.
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

        # Buffers for drone and ball states
        self.last_time: Optional[float] = None
        self.last_drone_state: Optional[DroneState] = None
        self.last_ball_state: Optional[BallState] = None
        self.last_policy_request: Optional[np.ndarray] = np.zeros((4,))
        self.drone_state_buffer: Deque[DroneState] = deque(maxlen=self.BUFFER_DRONE_HISTORY_SIZE)

    def on_event(self, event: Event):
        if isinstance(event, StopRequestEvent):
            self._handle_stop_request()
        elif isinstance(event, ArmRequestEvent):
            self._handle_arm_request()
        elif isinstance(event, PolicyStateUpdateEvent):
            self._handle_policy_state_update(t=event.t, drone_state=event.drone_state, ball_state=event.ball_state)
        else:
            raise ValueError("Unknown event type")

    def _handle_arm_request(self):
        if self.state in  (StateMachineState.STOPPED, StateMachineState.RECOVERY):
            #TODO: Check pre-arm conditions if needed
            self.state = StateMachineState.ARMED
            self.effects.go_to_origin()
            self.effects.reset_observation()
            self.effects.logging("SM armed.")
        else:
            self.effects.logging("Arm request ignored in current state: " + str(self.state))

    def _handle_stop_request(self):
        if self.state in (StateMachineState.ARMED, StateMachineState.RUNNING, StateMachineState.RECOVERY):
            self.state = StateMachineState.STOPPED
            self.effects.reset_observation()
            self.effects.logging("SM stopped.")
        else:
            self.effects.logging("Stop request ignored in current state: " + str(self.state))

    def _handle_policy_state_update(self, t: float, drone_state: DroneState, ball_state: BallState):

        self.last_time = t
        self.last_drone_state = drone_state
        self.last_ball_state = ball_state
        # --- Update drone state buffer ---
        with self._buf_lock:
            self.drone_state_buffer.appendleft(drone_state)
        
        # --- Push observation data ---
        observation_data = ObservationData(
            drone_state=drone_state,
            ball_state=ball_state,
            last_policy_request=self.last_policy_request
        )
        self.effects.push_observation(observation_data)
        
        # --- Transition based on state ---            
        drone_outside_safety_bounds = not utilities.is_within_bounds(drone_state.position, ALL_DRONE_POSITION_BOUNDS)
        if drone_outside_safety_bounds and self.state != StateMachineState.STOPPED:
            self.state = StateMachineState.STOPPED
            self.effects.logging(f"Drone outside safety bounds ({drone_state.position}). Stopping policy.", LoggingLevel.WARN)

        elif self.state == StateMachineState.ARMED:
            if self._check_pre_run_conditions():
                self.effects.reset_observation() #TODO: Check if this make sense to add 
                self.state = StateMachineState.RUNNING
                self.effects.logging("Ball detected within ARMED bounds. Transitioning to RUNNING state.", LoggingLevel.INFO)
                
        elif self.state == StateMachineState.RUNNING:
            if not self._check_running_conditions():
                # self.state = StateMachineState.RECOVERY #TODO: DEBUG
                self.state = StateMachineState.STOPPED #TODO: DEBUG
                self.effects.logging("Running conditions not met. Transitioning to RECOVERY state.", LoggingLevel.WARN)

        elif self.state == StateMachineState.RECOVERY:
            if not self._check_recovery_conditions():
                self.state = StateMachineState.STOPPED
                self.effects.logging("Recovery conditions not met. Transitioning to STOPPED state.", LoggingLevel.WARN)

        # --- Execute effects based on state ---
        if self.state == StateMachineState.STOPPED:
            pass
            
        elif self.state == StateMachineState.RUNNING:
            self.last_policy_request = self.effects.run_bounce_control()
        
        elif self.state == StateMachineState.RECOVERY:
            recovery_control = self.effects.run_recovery_control()
            if recovery_control is not None:
                self.last_policy_request = recovery_control
            else:
                self.last_policy_request = np.zeros((4,))


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
    
    def _validate_pre_arm_conditions(self) -> bool:
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
                if drone_state.time < current_time - self.START_CHECK_WINDOW_DURATION:
                    has_enough_duration = True
                    break
                num_samples_within_timeframe += 1
                
                if not utilities.is_within_bounds(drone_state.position, TO_ARM_DRONE_POSITION_BOUNDS):
                    self.effects.logging(f"Drone position out of bounds: {drone_state.position}", LoggingLevel.WARN)
                    failed_checks = True
                    break

                if not utilities.is_within_bounds(drone_state.velocity, TO_ARM_DRONE_VELOCITY_BOUNDS):
                    self.effects.logging(
                        f"Drone is not stationary (Linear Velocity = [{drone_state.velocity[0]}, {drone_state.velocity[1]}, {drone_state.velocity[2]}])",
                        LoggingLevel.WARN
                    )
                    failed_checks = True
                    break
                
                if not utilities.is_within_bounds(drone_state.body_rate, TO_ARM_DRONE_BODY_RATE_BOUNDS):
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
    
    def _check_pre_run_conditions(self) -> bool:
        # Perform necessary checks before starting the policy:
        # - The ball is within the defined bounds
        
        if self.last_ball_state is None:
            return False

        if not utilities.is_within_bounds(self.last_ball_state.position, ARMED_BALL_POSITION_TO_RUNNING_BOUNDS):
            return False

        return True
    
    def _check_running_conditions(self) -> bool:
        # Perform necessary checks while running the policy:
        # - The ball is within the defined bounds
        # - The drone is within the defined bounds
        # - The drone angle is within the defined bounds
        
        if self.last_ball_state is None or self.last_drone_state is None:
            self.effects.logging("Last ball or drone state is None.", LoggingLevel.WARN)
            return False

        if not utilities.is_within_bounds(self.last_ball_state.position, RUNNING_BALL_POSITION_BOUNDS):
            self.effects.logging("Ball position out of bounds.", LoggingLevel.WARN)
            return False

        if not utilities.is_within_bounds(self.last_drone_state.position, RUNNING_DRONE_POSITION_BOUNDS):
            self.effects.logging("Drone position out of bounds.", LoggingLevel.WARN)
            return False

        if self._drone_angle_from_vertical(self.last_drone_state) >= RUNNING_DRONE_MAX_ANGLE:
            self.effects.logging("Drone angle from vertical exceeds maximum allowed angle.", LoggingLevel.WARN)
            return False

        return True

    def _check_recovery_conditions(self) -> bool:
        # Perform necessary checks while in recovery mode:
        # - The drone is within the defined bounds
        # - The drone angle is within the defined bounds
        
        if self.last_drone_state is None:
            return False

        if not utilities.is_within_bounds(self.last_drone_state.position, ALL_DRONE_POSITION_BOUNDS):
            return False

        if self._drone_angle_from_vertical(self.last_drone_state) >= RECOVERY_DRONE_MAX_ANGLE:
            return False

        return True


