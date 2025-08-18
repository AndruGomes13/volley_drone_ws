import numpy as np
from dataclasses import dataclass, field
from scipy.spatial.transform import Rotation
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from copy import deepcopy
import plotly.io as pio
pio.renderers.default = "browser"   # opens in your default web browser
from ESEKF import ViconESEKFConstOmega, rts_smooth, NominalState, NoiseParams, FilterHistoryPoint, copy_state

# ==============================================================================
# ---------- Test Trajectory Generation and Plotting ----------
# ==============================================================================

def generate_helical_trajectory(duration: float, dt: float, radius: float, pitch: float, revs: float):
    """Generates a smooth helical trajectory."""
    t = np.arange(0, duration, dt)
    n_steps = len(t)
    
    # Pre-allocate arrays
    p = np.zeros((n_steps, 3))
    v = np.zeros((n_steps, 3))
    q = np.zeros((n_steps, 4))
    w = np.zeros((n_steps, 3))

    # Trajectory parameters
    yaw_rate = (revs * 2 * np.pi) / duration
    vz = (revs * pitch) / duration
    
    # Initial state
    current_rot = Rotation.identity()
    q[0] = current_rot.as_quat()
    
    for k in range(1, n_steps):
        time = t[k]
        angle = yaw_rate * time
        
        # Position and Velocity (derivatives of each other)
        p[k] = [radius * np.cos(angle), radius * np.sin(angle), vz * time]
        v[k] = [-radius * yaw_rate * np.sin(angle), radius * yaw_rate * np.cos(angle), vz]
        
        # Angular Velocity (constant in body frame, which is rotating with the helix)
        w[k] = [0, 0, yaw_rate]
        
        # Orientation
        rot_change = Rotation.from_rotvec(w[k] * dt)
        current_rot = rot_change * current_rot
        q[k] = current_rot.as_quat()
        
    return t, p, v, q, w


def plot_results(t, truth, filtered, smoothed):
    """Plots ground truth, filtered, and smoothed results using Plotly."""
    p_true, v_true, _, w_true = truth
    p_filt, v_filt, _, w_filt = filtered
    p_smth, v_smth, _, w_smth = smoothed

    fig = make_subplots(
        rows=2, cols=2,
        specs=[[{'type': 'scatter3d'}, {'type': 'xy'}], [{'type': 'xy'}, {'type': 'xy'}]],
        subplot_titles=("Position", "Velocity", "Angular Velocity", "Velocity RMS Error")
    )

    # 1. 3D Position Plot
    fig.add_trace(go.Scatter3d(x=p_true[:,0], y=p_true[:,1], z=p_true[:,2], mode='lines', name='Truth', line=dict(color='black')), row=1, col=1)
    fig.add_trace(go.Scatter3d(x=p_filt[:,0], y=p_filt[:,1], z=p_filt[:,2], mode='lines', name='Filtered', line=dict(color='blue', dash='dash')), row=1, col=1)
    fig.add_trace(go.Scatter3d(x=p_smth[:,0], y=p_smth[:,1], z=p_smth[:,2], mode='lines', name='Smoothed', line=dict(color='red')), row=1, col=1)

    # 2. Velocity Plot
    colors = ['blue', 'green', 'purple']
    for i, axis in enumerate(['Vx', 'Vy', 'Vz']):
        fig.add_trace(go.Scatter(x=t, y=v_true[:,i], mode='lines', name=f'Truth {axis}', line=dict(color='black', width=1)), row=1, col=2)
        fig.add_trace(go.Scatter(x=t, y=v_filt[:,i], mode='lines', name=f'Filt {axis}', line=dict(color=colors[i], dash='dash')), row=1, col=2)
        fig.add_trace(go.Scatter(x=t, y=v_smth[:,i], mode='lines', name=f'Smth {axis}', line=dict(color=colors[i])), row=1, col=2)
    
    # 3. Angular Velocity Plot
    for i, axis in enumerate(['Wx', 'Wy', 'Wz']):
        fig.add_trace(go.Scatter(x=t, y=w_true[:,i], mode='lines', name=f'Truth {axis}', line=dict(color='black', width=1)), row=2, col=1)
        fig.add_trace(go.Scatter(x=t, y=w_filt[:,i], mode='lines', name=f'Filt {axis}', line=dict(color=colors[i], dash='dash')), row=2, col=1)
        fig.add_trace(go.Scatter(x=t, y=w_smth[:,i], mode='lines', name=f'Smth {axis}', line=dict(color=colors[i])), row=2, col=1)

    # 4. Velocity Error Plot
    err_filt = np.linalg.norm(v_true - v_filt, axis=1)
    err_smth = np.linalg.norm(v_true - v_smth, axis=1)
    fig.add_trace(go.Scatter(x=t, y=err_filt, mode='lines', name='Filtered Error', line=dict(color='blue')), row=2, col=2)
    fig.add_trace(go.Scatter(x=t, y=err_smth, mode='lines', name='Smoothed Error', line=dict(color='red')), row=2, col=2)

    # Update layout
    fig.update_layout(
        title_text="ESEKF and RTS Smoother Performance",
        height=900,
        scene=dict(xaxis_title='X (m)', yaxis_title='Y (m)', zaxis_title='Z (m)', aspectratio=dict(x=1, y=1, z=0.5)),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    fig.update_xaxes(title_text="Time (s)", row=1, col=2)
    fig.update_yaxes(title_text="Velocity (m/s)", row=1, col=2)
    fig.update_xaxes(title_text="Time (s)", row=2, col=1)
    fig.update_yaxes(title_text="Ang. Vel (rad/s)", row=2, col=1)
    fig.update_xaxes(title_text="Time (s)", row=2, col=2)
    fig.update_yaxes(title_text="RMS Error (m/s)", row=2, col=2)
    
    fig.show()

# ==============================================================================
# ---------- Main Execution Block ----------
# ==============================================================================

if __name__ == "__main__":
    np.set_printoptions(precision=4, suppress=True)
    rng = np.random.default_rng(42)

    # --- 1. Simulation Setup ---
    DT = 0.01  # 100 Hz
    DURATION = 10.0  # seconds
    
    t, p_true, v_true, q_true, w_true = generate_helical_trajectory(
        duration=DURATION, dt=DT, radius=1.0, pitch=0.5, revs=2
    )
    n_steps = len(t)

    # --- 2. EKF Initialization ---
    noise_params = NoiseParams(
        sigma_a=0.1,           # Process noise on acceleration (m/s^2/√Hz)
        sigma_alpha=0.01,       # Process noise on ang. acceleration (rad/s^2/√Hz)
        sigma_p_meas=0.02,    # Vicon position measurement noise (m)
        sigma_theta_meas=0.01  # Vicon orientation measurement noise (rad)
    )

    
    # Initialize EKF at the true initial state
    initial_state = NominalState(p=p_true[0], v=v_true[0], q=q_true[0], w=w_true[0])
    ekf = ViconESEKFConstOmega(state=initial_state, noise=noise_params)

    # --- 3. Run Filter and Store History ---
    history: list[FilterHistoryPoint] = []
    print(f"Running simulation for {n_steps} steps...")

    for k in range(n_steps):
        # Predict step
        ekf.predict(DT)
        
        # Store predicted state
        x_pred = copy_state(ekf.x)
        P_pred = ekf.cov.P.copy()
        F      = ekf.F_discrete(DT)  # OK to recompute here *before* update
        
        # Generate noisy measurement from ground truth
        p_noise = rng.normal(0, noise_params.sigma_p_meas, 3)
        rot_noise = Rotation.from_rotvec(rng.normal(0, noise_params.sigma_theta_meas, 3))
        
        meas_p = p_true[k] + p_noise
        meas_q = (rot_noise * Rotation.from_quat(q_true[k])).as_quat()
        
        # Update step
        ekf.update(meas_p, meas_q)
        
        x_upd = copy_state(ekf.x)
        P_upd = ekf.cov.P.copy()
        
        # Store results for this step
        history.append(FilterHistoryPoint(
            t=t[k], dt=DT, F=F,
            x_pred=x_pred, P_pred=P_pred,
            x_upd=x_upd, P_upd=P_upd
        ))

    print("Forward pass (EKF) complete.")

    # --- 4. Run RTS Smoother ---
    smoothed_states = rts_smooth(history)
    print("Backward pass (RTS Smoother) complete.")

    # --- 5. Extract and Plot Results ---
    # Extract filtered states from history
    p_filt = np.array([h.x_upd.p for h in history])
    v_filt = np.array([h.x_upd.v for h in history])
    q_filt = np.array([h.x_upd.q for h in history])
    w_filt = np.array([h.x_upd.w for h in history])

    # Extract smoothed states
    p_smth = np.array([s.p for s in smoothed_states])
    v_smth = np.array([s.v for s in smoothed_states])
    q_smth = np.array([s.q for s in smoothed_states])
    w_smth = np.array([s.w for s in smoothed_states])

    plot_results(
        t,
        truth=(p_true, v_true, q_true, w_true),
        filtered=(p_filt, v_filt, q_filt, w_filt),
        smoothed=(p_smth, v_smth, q_smth, w_smth)
    )
    
    # --- 6. NEW: Statistical Analysis ---
    def calculate_rmse(est, truth):
        return np.sqrt(np.mean(np.sum((est - truth)**2, axis=1)))

    rmse_p_filt = calculate_rmse(p_filt, p_true)
    rmse_p_smth = calculate_rmse(p_smth, p_true)
    rmse_v_filt = calculate_rmse(v_filt, v_true)
    rmse_v_smth = calculate_rmse(v_smth, v_true)
    rmse_w_filt = calculate_rmse(w_filt, w_true)
    rmse_w_smth = calculate_rmse(w_smth, w_true)
    
    def improvement(filt, smth):
        return (1 - (smth / filt)) * 100 if filt > 0 else 0

    print("\n" + "="*50)
    print(" " * 15 + "PERFORMANCE ANALYSIS")
    print("="*50)
    print(f"{'Metric':<15} | {'Filtered (EKF)':<15} | {'Smoothed (RTS)':<15} | {'Improvement':<15}")
    print("-"*50)
    print(f"{'Pos RMSE (m)':<15} | {rmse_p_filt:<15.4f} | {rmse_p_smth:<15.4f} | {improvement(rmse_p_filt, rmse_p_smth):>14.2f}%")
    print(f"{'Vel RMSE (m/s)':<15} | {rmse_v_filt:<15.4f} | {rmse_v_smth:<15.4f} | {improvement(rmse_v_filt, rmse_v_smth):>14.2f}%")
    print(f"{'AngVel RMSE (rad/s)':<15} | {rmse_w_filt:<15.4f} | {rmse_w_smth:<15.4f} | {improvement(rmse_w_filt, rmse_w_smth):>14.2f}%")
    print("="*50 + "\n")
