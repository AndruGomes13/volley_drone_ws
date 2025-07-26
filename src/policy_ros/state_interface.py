from collections import deque

from typing import Deque, Optional
from typing_extensions import Self
import numpy as np
from dataclasses import dataclass

"""
NOTE: This can be optimized further by using JAX's to it's full potential.
Not doing it now to keep the code simple and readable.
"""

@dataclass
class DroneState:
    time:float
    position: np.ndarray
    velocity: np.ndarray
    orientation_wxyz: np.ndarray
    body_rate: np.ndarray
    angular_acceleration: np.ndarray
    gyro_bias: np.ndarray
    
@dataclass
class DroneStateHistoryNumpy:
    time: np.ndarray
    position: np.ndarray
    velocity: np.ndarray
    orientation_wxyz: np.ndarray
    body_rate: np.ndarray
    angular_acceleration: np.ndarray
    gyro_bias: np.ndarray

    @classmethod
    def from_drone_state_history(cls, drone_state_history: Deque[DroneState], history_length: Optional[int]= None) -> Self:
        """
        Convert a deque of DroneState to DroneStateHistoryNumpy.
        """
        if history_length is None:
            history_length = len(drone_state_history)
            
        if history_length > len(drone_state_history):
            raise ValueError(f"Requested history length {history_length} exceeds the available drone state history length {len(drone_state_history)}.")

        drone_state_history_list = list(drone_state_history)[:history_length]

        time = np.array([state.time for state in drone_state_history_list])
        position = np.array([state.position for state in drone_state_history_list])
        velocity = np.array([state.velocity for state in drone_state_history_list])
        orientation_wxyz = np.array([state.orientation_wxyz for state in drone_state_history_list])
        body_rate = np.array([state.body_rate for state in drone_state_history_list])
        angular_acceleration = np.array([state.angular_acceleration for state in drone_state_history_list])
        gyro_bias = np.array([state.gyro_bias for state in drone_state_history_list])

        return cls(
            time=time,
            position=position,
            velocity=velocity,
            orientation_wxyz=orientation_wxyz,
            body_rate=body_rate,
            angular_acceleration=angular_acceleration,
            gyro_bias=gyro_bias
        )

@dataclass
class BallState:
    time:float
    position: np.ndarray
    velocity: np.ndarray
    
@dataclass
class BallStateHistoryNumpy:
    time: np.ndarray
    position: np.ndarray
    velocity: np.ndarray

    @classmethod
    def from_ball_state_history(cls, ball_state_history: Deque[BallState], history_length: Optional[int]= None) -> Self:
        """
        Convert a deque of BallState to BallStateHistoryNumpy.
        """
        if history_length is None:
            history_length = len(ball_state_history)
            
        if history_length > len(ball_state_history):
            raise ValueError(f"Requested history length {history_length} exceeds the available ball state history length {len(ball_state_history)}.")

        ball_state_history_list = list(ball_state_history)[:history_length]

        time = np.array([state.time for state in ball_state_history_list])
        position = np.array([state.position for state in ball_state_history_list])
        velocity = np.array([state.velocity for state in ball_state_history_list])

        return cls(
            time=time,
            position=position,
            velocity=velocity
        )

@dataclass
class StateHistory:
    """A class to hold the history of states in an observation pipeline."""
    drone_state_history: Deque[DroneState]  # Assuming DroneState is defined elsewhere
    action_history: Deque[np.ndarray]
    ball_state_history: Deque[BallState]

@dataclass
class StateHistoryNumpy:
    """A class to hold the history of states in an observation pipeline using JAX arrays."""
    drone_state_history: DroneStateHistoryNumpy
    action_history: np.ndarray
    ball_state_history: BallStateHistoryNumpy

    @classmethod
    def from_state_history(cls, state_history: StateHistory, drone_history_length: Optional[int] = None, ball_history_length: Optional[int] = None, action_history_length: Optional[int] = None) -> Self:
        drone_state_history_numpy = DroneStateHistoryNumpy.from_drone_state_history(state_history.drone_state_history, drone_history_length)
        ball_state_history_numpy = BallStateHistoryNumpy.from_ball_state_history(state_history.ball_state_history, ball_history_length)

        action_history = np.array(list(state_history.action_history)[:action_history_length]) if action_history_length else np.array(state_history.action_history)

        return cls(
            drone_state_history=drone_state_history_numpy,
            action_history=action_history,
            ball_state_history=ball_state_history_numpy
        )
    
        
if __name__ == "__main__":
    """
    Benchmark for converting StateHistory to StateHistoryNumpy.
    
    """
    import time
    from collections import deque

    # ----- parameters -----
    N_DRONE   = 200   # how many drone states
    N_BALL    = 200    # how many ball   states
    N_ACTIONS = 20     # how many actions
    np.random.seed(0)

    # ----- helpers -----
    def random_drone_state(t: float) -> DroneState:
        return DroneState(
            time=t,
            position=np.random.randn(3),
            velocity=np.random.randn(3),
            orientation_wxyz=np.random.randn(4),
            body_rate=np.random.randn(3),
            angular_acceleration=np.random.randn(3),
            gyro_bias=np.random.randn(3),
        )

    def random_ball_state(t: float) -> BallState:
        return BallState(
            time=t,
            position=np.random.randn(3),
            velocity=np.random.randn(3),
        )    
    
    # --- Run the main benchmark ---
    t_sum = 0
    IT = 1000  # Number of iterations for the full benchmark
    for i in range(IT):
        state_hist = StateHistory(
            drone_state_history=deque(random_drone_state(float(i) * 0.01) for i in range(N_DRONE)),
            ball_state_history=deque(random_ball_state(float(i) * 0.01) for i in range(N_BALL)),
            action_history=deque(np.random.randn(4) for _ in range(N_ACTIONS)),
        )
        t_s = time.perf_counter()
        full_jax  = StateHistoryJax.from_state_history(state_hist)
        t_f = time.perf_counter()
        
        t_sum += (t_f - t_s) if i > 0 else 0  # Skip the first iteration to avoid warm-up time

    print(f"Full   conversion: {(t_sum / IT)*1e3:.2f} ms")