import numpy as np
from scipy.spatial.transform import Rotation
from dataclasses import dataclass
from typing import Any, Optional, Tuple

from rts_filter import ModelFns  # Adjust import as needed

@dataclass
class RigidBodyState:
    """
    Nominal state for a rigid body (translation only):
    - position: 3D position vector
    - velocity: 3D velocity vector  
    """
    position: np.ndarray        # shape (3,)
    velocity: np.ndarray        # shape (3,)

def create_rigid_body_model(
    process_noise_std: dict,
    measurement_noise_std: dict,
) -> 'ModelFns':
    """
    Create ModelFns for rigid body dynamics (translation only).
    
    Error state is 6D: [pos_err(3), vel_err(3)]
    
    Args:
        process_noise_std: dict with key 'accel' for process noise
        measurement_noise_std: dict with key 'position' for measurement noise
    """
    
    def propagate_nominal(nom_state: RigidBodyState, x_err: np.ndarray, dt: float, u: Optional[Any]) -> RigidBodyState:
        """Propagate nominal state using simple rigid body dynamics."""
        
        # Extract current state (no error correction here - that's done in compose_state)
        pos = nom_state.position.copy()
        vel = nom_state.velocity.copy()
        
        # Rigid body dynamics (constant velocity model)
        # Position: p_k+1 = p_k + v_k * dt
        # Velocity: v_k+1 = v_k (constant velocity assumption)
        
        new_pos = pos + vel * dt
        new_vel = vel.copy()
        
        return RigidBodyState(
            position=new_pos,
            velocity=new_vel,
        )
    
    def A_matrix(k: int, dt: float, nom_state: RigidBodyState, x_err: np.ndarray, u: Optional[Any]) -> np.ndarray:
        """State transition matrix for error state (6x6)."""
        A = np.eye(6)
        
        # Position error propagation: δp_{k+1} = δp_k + δv_k * dt
        A[0:3, 3:6] = np.eye(3) * dt
        
        return A
    
    def Q_matrix(k: int, dt: float, nom_state: RigidBodyState, x_err: np.ndarray, u: Optional[Any]) -> np.ndarray:
        """Process noise covariance matrix (6x6)."""
        Q = np.zeros((6, 6))
        
        # Process noise on accelerations affects position and velocity
        accel_var = process_noise_std['accel']**2
        
        # Position noise from acceleration integration: σ_p^2 = σ_a^2 * dt^4 / 4
        Q[0:3, 0:3] = np.eye(3) * accel_var * dt**4 / 4
        
        # Cross-correlation position-velocity: σ_pv = σ_a^2 * dt^3 / 2  
        Q[0:3, 3:6] = np.eye(3) * accel_var * dt**3 / 2
        Q[3:6, 0:3] = np.eye(3) * accel_var * dt**3 / 2

        # Velocity noise from acceleration: σ_v^2 = σ_a^2 * dt^2
        Q[3:6, 3:6] = np.eye(3) * accel_var * dt**2

        return Q
    
    def predict_measurement(k: int, nom_state: RigidBodyState) -> np.ndarray:
        """Predict measurement: position only [position(3)]."""
        return nom_state.position.copy()
    
    def H_matrix(k: int, nom_state: RigidBodyState) -> np.ndarray:
        """Measurement matrix: observes position only (3x6)."""
        H = np.zeros((3, 6))
        
        # Position measurement directly observes position error
        H[0:3, 0:3] = np.eye(3)        
        return H
    
    def R_matrix(k: int) -> np.ndarray:
        """Measurement noise covariance (3x3)."""
        return np.eye(3) * measurement_noise_std['position']**2
    
    def residual_measurement(y_meas: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        """Compute measurement residual."""
        return y_meas - y_pred
    
    def compose_state(nom_state: RigidBodyState, x_err: np.ndarray) -> RigidBodyState:
        """Apply error state to nominal state."""
        pos_err = x_err[:3]
        vel_err = x_err[3:6]
        
        # Apply errors to nominal state
        new_pos = nom_state.position + pos_err
        new_vel = nom_state.velocity + vel_err
        
        return RigidBodyState(
            position=new_pos,
            velocity=new_vel,
        )
    
    def reset_error_state(x_err: np.ndarray, P_err: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Reset error state to zero after applying to nominal state."""
        x_reset = np.zeros_like(x_err)
        P_reset = P_err.copy()
        
        return x_reset, P_reset
    
    return ModelFns(
        propagate_nominal=propagate_nominal,
        A=A_matrix,
        Q=Q_matrix,
        predict_meas=predict_measurement,
        H=H_matrix,
        R=R_matrix,
        residual_meas=residual_measurement,
        compose=compose_state,
        reset_error=reset_error_state
    )

# Example usage function
def create_example_rigid_body_filter():
    """Create an example rigid body filter setup for Vicon data."""
    
    # Define noise parameters (tune these based on your Vicon system)
    process_noise = {
        'accel': 0.1,  # m/s^2 - acceleration noise std
    }
    
    measurement_noise = {
        'position': 0.0005,     # m - Vicon position accuracy (very good) 
    }
    
    # Create the model
    model = create_rigid_body_model(process_noise, measurement_noise)
    
    # Filter setup
    n_states = 6  # [pos_err(3), vel_err(3)]
    
    # Initial state
    initial_position = np.array([0.0, 0.0, 1.0])  # 1m above ground
    initial_velocity = np.array([0.0, 0.0, 0.0])
    
    nom0 = RigidBodyState(
        position=initial_position,
        velocity=initial_velocity,
    )
    
    # Initial error state (zeros)
    x0 = np.zeros(n_states)
    
    # Initial covariance (tune based on initial uncertainty)
    P0 = np.diag([
        0.01, 0.01, 0.01,  # position uncertainty (1cm std)
        0.1, 0.1, 0.1,     # velocity uncertainty (0.1 m/s std)
    ])
    
    return model, n_states, x0, P0, nom0