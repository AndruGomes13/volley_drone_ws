import numpy as np
from scipy.spatial.transform import Rotation
from dataclasses import dataclass
from typing import Any, Optional, Tuple

from rts_filter import ModelFns  # Adjust import as needed

@dataclass
class RigidBodyState:
    """
    Nominal state for a rigid body:
    - position: 3D position vector
    - quaternion: orientation as scipy Rotation object
    - velocity: 3D velocity vector  
    - angular_velocity: 3D angular velocity vector (body frame)
    """
    position: np.ndarray        # shape (3,)
    quaternion: Rotation        # scipy Rotation object
    velocity: np.ndarray        # shape (3,)
    angular_velocity: np.ndarray # shape (3,) in body frame

def create_rigid_body_model(
    process_noise_std: dict,
    measurement_noise_std: dict,
    gravity: np.ndarray = np.array([0, 0, -9.81])
) -> 'ModelFns':
    """
    Create ModelFns for rigid body dynamics.
    
    Error state is 12D: [pos_err(3), att_err(3), vel_err(3), gyro_err(3)]
    
    Args:
        process_noise_std: dict with keys 'accel', 'gyro' for process noise
        measurement_noise_std: dict with keys 'position', 'orientation' 
        gravity: gravity vector in world frame
    """
    
    def propagate_nominal(nom_state: RigidBodyState, x_err: np.ndarray, dt: float, u: Optional[Any]) -> RigidBodyState:
        """Propagate nominal state using simple rigid body dynamics."""
        
        # Extract current state
        pos = nom_state.position.copy()
        quat = nom_state.quaternion
        vel = nom_state.velocity.copy()
        omega = nom_state.angular_velocity.copy()
        
        # Apply error corrections (error state is applied here for prediction)
        pos_err, att_err, vel_err, omega_err = x_err[:3], x_err[3:6], x_err[6:9], x_err[9:12]
        
        pos += pos_err
        vel += vel_err
        omega += omega_err
        
        # Apply attitude error (small angle approximation)
        if np.linalg.norm(att_err) > 1e-8:
            delta_quat = Rotation.from_rotvec(att_err)
            quat = delta_quat * quat
        
        # Rigid body dynamics
        # Position: p_k+1 = p_k + v_k * dt + 0.5 * a_k * dt^2
        # Velocity: v_k+1 = v_k + a_k * dt
        # where a_k includes gravity
        
        accel = gravity  # Simple model: only gravity (could add control inputs here)
        
        new_pos = pos + vel * dt + 0.5 * accel * dt**2
        new_vel = vel + accel * dt
        
        # Orientation: integrate angular velocity
        if np.linalg.norm(omega) > 1e-8:
            # Convert body-frame angular velocity to world frame rotation
            omega_world = quat.apply(omega)  # Transform to world frame
            delta_angle = omega_world * dt
            delta_quat = Rotation.from_rotvec(delta_angle)
            new_quat = delta_quat * quat
        else:
            new_quat = quat
            
        # Angular velocity remains constant in this simple model
        new_omega = omega
        
        return RigidBodyState(
            position=new_pos,
            quaternion=new_quat,
            velocity=new_vel,
            angular_velocity=new_omega
        )
    
    def A_matrix(k: int, dt: float, nom_state: RigidBodyState, x_err: np.ndarray, u: Optional[Any]) -> np.ndarray:
        """State transition matrix for error state (12x12)."""
        A = np.eye(12)
        
        # Position error propagation: δp_{k+1} = δp_k + δv_k * dt
        A[0:3, 6:9] = np.eye(3) * dt
        
        # Attitude error propagation (first-order approximation)
        # δθ_{k+1} ≈ δθ_k - [ω×] * δθ_k * dt + δω_k * dt
        omega = nom_state.angular_velocity
        omega_cross = skew_symmetric(omega)
        A[3:6, 3:6] = np.eye(3) - omega_cross * dt
        A[3:6, 9:12] = np.eye(3) * dt
        
        # Velocity and angular velocity errors are unchanged in this simple model
        # δv_{k+1} = δv_k  (could add coupling from attitude errors in more complex model)
        # δω_{k+1} = δω_k
        
        return A
    
    def Q_matrix(k: int, dt: float, nom_state: RigidBodyState, x_err: np.ndarray, u: Optional[Any]) -> np.ndarray:
        """Process noise covariance matrix (12x12)."""
        Q = np.zeros((12, 12))
        
        # Process noise on accelerations affects position and velocity
        accel_var = process_noise_std['accel']**2
        
        # Position noise from acceleration integration: σ_p^2 = σ_a^2 * dt^4 / 4
        Q[0:3, 0:3] = np.eye(3) * accel_var * dt**4 / 4
        
        # Cross-correlation position-velocity: σ_pv = σ_a^2 * dt^3 / 2  
        Q[0:3, 6:9] = np.eye(3) * accel_var * dt**3 / 2
        Q[6:9, 0:3] = np.eye(3) * accel_var * dt**3 / 2
        
        # Velocity noise from acceleration: σ_v^2 = σ_a^2 * dt^2
        Q[6:9, 6:9] = np.eye(3) * accel_var * dt**2
        
        # Angular velocity noise affects attitude
        gyro_var = process_noise_std['gyro']**2
        
        # Attitude noise from angular velocity integration: σ_θ^2 = σ_ω^2 * dt^2
        Q[3:6, 3:6] = np.eye(3) * gyro_var * dt**2
        
        # Angular velocity noise (random walk)
        Q[9:12, 9:12] = np.eye(3) * gyro_var * dt
        
        return Q
    
    def predict_measurement(k: int, nom_state: RigidBodyState) -> np.ndarray:
        """Predict measurement: [position(3), rotation_vector(3)]."""
        pos = nom_state.position
        att_rotvec = nom_state.quaternion.as_rotvec()  # Rotation vector representation
        return np.concatenate([pos, att_rotvec])
    
    def H_matrix(k: int, nom_state: RigidBodyState) -> np.ndarray:
        """Measurement matrix: observes position and attitude (6x12)."""
        H = np.zeros((6, 12))
        
        # Position measurement directly observes position error
        H[0:3, 0:3] = np.eye(3)
        
        # Attitude measurement observes attitude error directly
        # We use rotation vector representation for attitude errors
        H[3:6, 3:6] = np.eye(3)
        
        return H
    
    def R_matrix(k: int) -> np.ndarray:
        """Measurement noise covariance (6x6)."""
        R = np.zeros((6, 6))
        R[0:3, 0:3] = np.eye(3) * measurement_noise_std['position']**2
        R[3:6, 3:6] = np.eye(3) * measurement_noise_std['orientation']**2
        return R
    
    def residual_measurement(y_meas: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        """Compute measurement residual with proper quaternion handling."""
        pos_residual = y_meas[:3] - y_pred[:3]
        
        # Convert quaternion measurement to rotation vector for residual
        if len(y_meas) == 7:  # Input is [pos(3), quat(4)]
            q_meas = Rotation.from_quat(y_meas[3:7])  # [x, y, z, w]
            att_meas_rotvec = q_meas.as_rotvec()
        else:  # Input is already [pos(3), rotvec(3)]
            att_meas_rotvec = y_meas[3:6]
        
        # Attitude residual in rotation vector space
        att_residual = att_meas_rotvec - y_pred[3:6]
        
        return np.concatenate([pos_residual, att_residual])
    
    def compose_state(nom_state: RigidBodyState, x_err: np.ndarray) -> RigidBodyState:
        """Apply error state to nominal state."""
        pos_err, att_err, vel_err, omega_err = x_err[:3], x_err[3:6], x_err[6:9], x_err[9:12]
        
        # Apply errors
        new_pos = nom_state.position + pos_err
        new_vel = nom_state.velocity + vel_err  
        new_omega = nom_state.angular_velocity + omega_err
        
        # Apply attitude error
        if np.linalg.norm(att_err) > 1e-8:
            delta_quat = Rotation.from_rotvec(att_err)
            new_quat = delta_quat * nom_state.quaternion
        else:
            new_quat = nom_state.quaternion
            
        return RigidBodyState(
            position=new_pos,
            quaternion=new_quat, 
            velocity=new_vel,
            angular_velocity=new_omega
        )
    
    def reset_error_state(x_err: np.ndarray, P_err: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Reset attitude errors to zero after applying to nominal state."""
        x_reset = x_err.copy()
        P_reset = P_err.copy()
        
        # Reset attitude errors (components 3:6) to zero
        x_reset[3:6] = 0.0
        
        # Option: could also reset position errors if desired
        # x_reset[0:3] = 0.0
        
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

def skew_symmetric(v: np.ndarray) -> np.ndarray:
    """Create skew-symmetric matrix from 3D vector."""
    return np.array([
        [ 0,    -v[2],  v[1]],
        [ v[2],  0,    -v[0]], 
        [-v[1],  v[0],  0   ]
    ])

# Example usage function
def create_example_rigid_body_filter():
    """Create an example rigid body filter setup for Vicon data."""
    
    # Define noise parameters (tune these based on your Vicon system)
    process_noise = {
        'accel': 0.1,  # m/s^2 - acceleration noise std
        'gyro': 0.01   # rad/s - angular velocity noise std  
    }
    
    measurement_noise = {
        'position': 0.001,     # m - Vicon position accuracy (very good)
        'orientation': 0.001   # rad - Vicon orientation accuracy  
    }
    
    # Create the model
    model = create_rigid_body_model(process_noise, measurement_noise)
    
    # Filter setup
    n_states = 12  # [pos(3), att(3), vel(3), omega(3)]
    
    # Initial state
    initial_position = np.array([0.0, 0.0, 1.0])  # 1m above ground
    initial_orientation = Rotation.from_euler('xyz', [0, 0, 0])  # identity
    initial_velocity = np.array([0.0, 0.0, 0.0])
    initial_angular_vel = np.array([0.0, 0.0, 0.0])
    
    nom0 = RigidBodyState(
        position=initial_position,
        quaternion=initial_orientation,
        velocity=initial_velocity,
        angular_velocity=initial_angular_vel
    )
    
    # Initial error state (usually zeros)
    x0 = np.zeros(n_states)
    
    # Initial covariance (tune based on initial uncertainty)
    P0 = np.diag([
        0.01, 0.01, 0.01,  # position uncertainty (1cm std)
        0.1, 0.1, 0.1,     # attitude uncertainty (0.1 rad std) 
        0.1, 0.1, 0.1,     # velocity uncertainty (0.1 m/s std)
        0.01, 0.01, 0.01   # angular velocity uncertainty (0.01 rad/s std)
    ])
    
    return model, n_states, x0, P0, nom0