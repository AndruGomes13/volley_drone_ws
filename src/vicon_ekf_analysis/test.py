import numpy as np
from dataclasses import dataclass, field
from scipy.spatial.transform import Rotation
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from copy import deepcopy
import plotly.io as pio
pio.renderers.default = "browser"   # opens in your default web browser
# ---------- Helpers ----------

def skew(w: np.ndarray) -> np.ndarray:
    wx, wy, wz = w
    return np.array([[0, -wz,  wy],
                     [wz,  0, -wx],
                     [-wy, wx,  0]])

def symmetrize(M: np.ndarray) -> np.ndarray:
    return 0.5 * (M + M.T)

# ---------- Data classes ----------

@dataclass
class NominalState:
    p: np.ndarray = field(default_factory=lambda: np.zeros(3))          # position (world)
    v: np.ndarray = field(default_factory=lambda: np.zeros(3))          # velocity (world)
    q: np.ndarray = field(default_factory=lambda: np.array([0,0,0,1]))  # quaternion [x,y,z,w] (world<-body)
    w: np.ndarray = field(default_factory=lambda: np.zeros(3))          # body rate (body frame)

    def rot(self) -> Rotation:
        return Rotation.from_quat(self.q)

    def normalize(self) -> None:
        self.q = self.q / np.linalg.norm(self.q)

def copy_state(x: NominalState) -> NominalState:
    return NominalState(p=x.p.copy(), v=x.v.copy(), q=x.q.copy(), w=x.w.copy())

@dataclass
class NoiseParams:
    # Continuous-time white noise stds (√PSD units)
    sigma_a: float = 0.3             # m/s^2 / √Hz  (drives v)
    sigma_alpha: float = 0.05        # rad/s^2 / √Hz (drives w)
    sigma_p_meas: float = 0.002      # m     (Vicon pos)
    sigma_theta_meas: float = 0.003  # rad   (Vicon orientation rotvec)

@dataclass
class Covariance:
    # Error-state covariance for [δp, δv, δθ, δω] (12x12)
    P: np.ndarray = field(default_factory=lambda: np.eye(12)*0.1)

@dataclass
class FilterHistoryPoint:
    """Stores the necessary filter results at a single timestep for smoothing."""
    t: float                      # Timestamp
    dt: float                     # dt to next step
    F: np.ndarray                 # State transition matrix used
    # --- Predicted values (x_{k|k-1}, P_{k|k-1})
    x_pred: NominalState
    P_pred: np.ndarray
    # --- Updated values (x_{k|k}, P_{k|k})
    x_upd: NominalState
    P_upd: np.ndarray
    
# ---------- Vicon-only ESEKF with constant body rate ----------

class ViconESEKFConstOmega:
    """
    Error-state EKF on SE(3) with state [p, v, q, w].
      - Process: constant v and constant body-rate w (with white acc & ang-acc driving noise)
      - Measurement: Vicon position + orientation
    Error state: δx = [δp, δv, δθ, δω] with left-invariant attitude error.
    """
    def __init__(self, state: NominalState | None = None,
                 cov: Covariance | None = None,
                 noise: NoiseParams | None = None):
        self.x = state if state is not None else NominalState()
        self.cov = cov if cov is not None else Covariance()
        self.noise = noise if noise is not None else NoiseParams()

        # Measurement Jacobian (constant): y = [p_meas - p; Log(R_meas R_nom^T)]
        self.H = np.zeros((6, 12))
        self.H[0:3, 0:3] = np.eye(3)   # δp -> pos residual
        self.H[3:6, 6:9] = np.eye(3)   # δθ -> attitude residual

    # ----- Model pieces -----

    def F_discrete(self, dt: float) -> np.ndarray:
        """
        Discrete error-state transition for CV + constant body rate:
          δp_k+1 = δp_k + δv_k dt
          δv_k+1 = δv_k
          δθ_k+1 = (I - [w×] dt) δθ_k + I dt δω_k
          δω_k+1 = δω_k
        """
        F = np.eye(12)
        F[0:3, 3:6] = np.eye(3) * dt                   # δp <- δv
        F[6:9, 6:9] = np.eye(3) - skew(self.x.w) * dt  # δθ ← (I - [w×] dt) δθ
        F[6:9, 9:12] = np.eye(3) * dt                  # δθ ← δw * dt
        return F

    def Qd_discrete(self, dt: float) -> np.ndarray:
        """
        Discrete process noise (independent per axis):
          For (δp, δv) driven by white accel q_a = sigma_a^2:
            Qpp = dt^3/3 q_a,  Qpv = dt^2/2 q_a,  Qvv = dt q_a
          For (δθ, δω) driven by white angular accel q_alpha = sigma_alpha^2:
            Qθθ = dt^3/3 q_alpha, Qθw = dt^2/2 q_alpha, Qww = dt q_alpha
        """
        q_a = self.noise.sigma_a**2
        q_al = self.noise.sigma_alpha**2

        Q = np.zeros((12, 12))

        # position/velocity block
        Qpp = (dt**3 / 3.0) * q_a * np.eye(3)
        Qpv = (dt**2 / 2.0) * q_a * np.eye(3)
        Qvv = (dt * q_a)       * np.eye(3)
        Q[0:3, 0:3] = Qpp
        Q[0:3, 3:6] = Qpv
        Q[3:6, 0:3] = Qpv
        Q[3:6, 3:6] = Qvv

        # attitude/body-rate block
        Qtt = (dt**3 / 3.0) * q_al * np.eye(3)
        Qtw = (dt**2 / 2.0) * q_al * np.eye(3)
        Qww = (dt * q_al)       * np.eye(3)
        Q[6:9, 6:9]   = Qtt
        Q[6:9, 9:12]  = Qtw
        Q[9:12, 6:9]  = Qtw
        Q[9:12, 9:12] = Qww

        return Q

    def nominal_integrate(self, dt: float) -> None:
        """
        p += v dt; q ← Exp(w dt) ⊗ q; v, w constant in nominal.
        """
        self.x.p = self.x.p + self.x.v * dt
        # integrate orientation with constant body-rate
        self.x.q = (Rotation.from_rotvec(self.x.w * dt) * self.x.rot()).as_quat()
        self.x.normalize()

    # ----- Predict/Update -----

    def predict(self, dt: float) -> None:
        if dt <= 0:
            return
        self.nominal_integrate(dt)
        F = self.F_discrete(dt)
        Qd = self.Qd_discrete(dt)
        self.cov.P = F @ self.cov.P @ F.T + Qd
        self.cov.P = symmetrize(self.cov.P)

    def residual(self, meas_p: np.ndarray, meas_q: np.ndarray) -> np.ndarray:
        """
        y = [p_meas - p_nom; Log(R_meas R_nomᵀ)]
        """
        y_p = meas_p - self.x.p
        Rm = Rotation.from_quat(meas_q)
        Rp = self.x.rot()
        y_theta = (Rm * Rp.inv()).as_rotvec()
        return np.concatenate([y_p, y_theta])

    def R_meas(self) -> np.ndarray:
        Rp = (self.noise.sigma_p_meas**2) * np.eye(3)
        Rth = (self.noise.sigma_theta_meas**2) * np.eye(3)
        R = np.zeros((6, 6))
        R[0:3, 0:3] = Rp
        R[3:6, 3:6] = Rth
        return R

    def inject_and_reset(self, dx: np.ndarray) -> None:
        """
        Inject δx into nominal and reset covariance. Left-invariant reset on attitude.
        """
        dp  = dx[0:3]
        dv  = dx[3:6]
        dth = dx[6:9]
        dw  = dx[9:12]

        # Inject into nominal
        self.x.p += dp
        self.x.v += dv
        self.x.q = (Rotation.from_rotvec(dth) * self.x.rot()).as_quat()
        self.x.w += dw
        self.x.normalize()

        # Reset Jacobian G:
        #   δθ' ≈ (I - 1/2 [dθ×]) δθ    (left-invariant)
        #   other blocks ~ I to first order
        G = np.eye(12)
        G[6:9, 6:9] = np.eye(3) - 0.5 * skew(dth)
        self.cov.P = G @ self.cov.P @ G.T
        self.cov.P = symmetrize(self.cov.P)

    def update(self, meas_p: np.ndarray, meas_q: np.ndarray) -> None:
        y = self.residual(meas_p, meas_q)
        Rm = self.R_meas()

        S = self.H @ self.cov.P @ self.H.T + Rm
        K = self.cov.P @ self.H.T @ np.linalg.solve(S, np.eye(6))
        dx = K @ y

        # Joseph form for numerical stability
        I = np.eye(12)
        KH = K @ self.H
        self.cov.P = (I - KH) @ self.cov.P @ (I - KH).T + K @ Rm @ K.T
        self.cov.P = symmetrize(self.cov.P)

        self.inject_and_reset(dx)

# ---------- RTS Smoother ----------
# Add this method inside your ViconESEKFConstOmega class

def rts_smooth(history: list[FilterHistoryPoint]) -> list[NominalState]:
    n_steps = len(history)
    if n_steps == 0: return []

    # Initialize smoothed trajectory with the final state from the forward pass
    smoothed_states = [None] * n_steps
    smoothed_states[-1] = history[-1].x_upd
    P_s = history[-1].P_upd.copy()

    # Iterate backwards from second-to-last state to the first
    for k in range(n_steps - 2, -1, -1):
        # Get required states and covariances from history
        x_f_k = history[k].x_upd         # Filtered state x_{k|k}
        P_f_k = history[k].P_upd         # Filtered covariance P_{k|k}
        x_p_k1 = history[k+1].x_pred     # Predicted state x_{k+1|k}
        P_p_k1 = history[k+1].P_pred     # Predicted covariance P_{k+1|k}
        F_k = history[k+1].F             # Transition matrix F_k
        x_s_k1 = smoothed_states[k+1]    # Smoothed state from future x_s_{k+1}

        # RTS Smoother Gain C_k = P_{k|k} * F_k^T * inv(P_{k+1|k})
        C_k = np.linalg.solve(P_p_k1, F_k @ P_f_k).T

        # --- Calculate residual between smoothed and predicted future states ---
        # This is the key step: x_s_{k+1} - x_{p,k+1}
        dp_res = x_s_k1.p - x_p_k1.p
        dv_res = x_s_k1.v - x_p_k1.v
        dth_res = (x_s_k1.rot() * x_p_k1.rot().inv()).as_rotvec()
        dw_res = x_s_k1.w - x_p_k1.w
        dx_residual = np.concatenate([dp_res, dv_res, dth_res, dw_res])

        # --- Update smoothed state and covariance ---
        # Correction: dx_s_k = C_k * dx_residual
        # New state: x_s_k = x_f_k + dx_s_k
        dx_s_k = C_k @ dx_residual
        
        # Inject correction into the filtered nominal state
        smoothed_k = NominalState()
        smoothed_k.p = x_f_k.p + dx_s_k[0:3]
        smoothed_k.v = x_f_k.v + dx_s_k[3:6]
        smoothed_k.q = (Rotation.from_rotvec(dx_s_k[6:9]) * x_f_k.rot()).as_quat()
        smoothed_k.w = x_f_k.w + dx_s_k[9:12]
        smoothed_k.normalize()
        smoothed_states[k] = smoothed_k

    return smoothed_states

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
        sigma_alpha=0.2,       # Process noise on ang. acceleration (rad/s^2/√Hz)
        sigma_p_meas=0.002,    # Vicon position measurement noise (m)
        sigma_theta_meas=0.001  # Vicon orientation measurement noise (rad)
    )
    noise_params = NoiseParams(sigma_a=0.5, sigma_alpha=0.2, sigma_p_meas=0.005, sigma_theta_meas=0.01)

    
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






# if __name__ == "__main__":
#     np.set_printoptions(precision=4, suppress=True)
#     rng = np.random.default_rng(42)

#     # --- 1. Simulation Setup ---
#     DT, DURATION = 0.01, 10.0
#     t, p_true, v_true, q_true, w_true = generate_helical_trajectory(
#         duration=DURATION, dt=DT, radius=1.0, pitch=0.5, revs=2
#     )
#     n_steps = len(t)

#     # --- 2. EKF Initialization ---
#     noise_params = NoiseParams(sigma_a=0.8, sigma_alpha=0.2, sigma_p_meas=0.005, sigma_theta_meas=0.01)
#     initial_state = NominalState(p=p_true[0], v=v_true[0], q=q_true[0], w=w_true[0])
#     ekf = ViconESEKFConstOmega(state=initial_state, noise=noise_params)

#     # --- 3. Run Filter and Store History ---
#     history: list[FilterHistoryPoint] = []
#     print(f"Running simulation for {n_steps} steps...")
#     for k in range(n_steps):
#         F = ekf.F_discrete(DT)
#         ekf.predict(DT)
#         x_pred, P_pred = ekf.x, ekf.cov.P
#         p_noise = rng.normal(0, noise_params.sigma_p_meas, 3)
#         rot_noise = Rotation.from_rotvec(rng.normal(0, noise_params.sigma_theta_meas, 3))
#         meas_p, meas_q = p_true[k] + p_noise, (rot_noise * Rotation.from_quat(q_true[k])).as_quat()
#         ekf.update(meas_p, meas_q)
#         history.append(FilterHistoryPoint(t=t[k], dt=DT, F=F, x_pred=x_pred, P_pred=P_pred, x_upd=ekf.x, P_upd=ekf.cov.P))
#     print("Forward pass (EKF) complete.")

#     # --- 4. Run RTS Smoother ---
#     smoothed_states = rts_smooth(history)
#     print("Backward pass (RTS Smoother) complete.")

#     # --- 5. Extract and Plot Results ---
#     p_filt = np.array([h.x_upd.p for h in history])
#     v_filt = np.array([h.x_upd.v for h in history])
#     w_filt = np.array([h.x_upd.w for h in history])
#     p_smth = np.array([s.p for s in smoothed_states])
#     v_smth = np.array([s.v for s in smoothed_states])
#     w_smth = np.array([s.w for s in smoothed_states])

#     plot_results(t, (p_true, v_true, q_true, w_true), (p_filt, v_filt, None, w_filt), (p_smth, v_smth, None, w_smth))

#     # --- 6. NEW: Statistical Analysis ---
#     def calculate_rmse(est, truth):
#         return np.sqrt(np.mean(np.sum((est - truth)**2, axis=1)))

#     rmse_p_filt = calculate_rmse(p_filt, p_true)
#     rmse_p_smth = calculate_rmse(p_smth, p_true)
#     rmse_v_filt = calculate_rmse(v_filt, v_true)
#     rmse_v_smth = calculate_rmse(v_smth, v_true)
#     rmse_w_filt = calculate_rmse(w_filt, w_true)
#     rmse_w_smth = calculate_rmse(w_smth, w_true)
    
#     def improvement(filt, smth):
#         return (1 - (smth / filt)) * 100 if filt > 0 else 0

#     print("\n" + "="*50)
#     print(" " * 15 + "PERFORMANCE ANALYSIS")
#     print("="*50)
#     print(f"{'Metric':<15} | {'Filtered (EKF)':<15} | {'Smoothed (RTS)':<15} | {'Improvement':<15}")
#     print("-"*50)
#     print(f"{'Pos RMSE (m)':<15} | {rmse_p_filt:<15.4f} | {rmse_p_smth:<15.4f} | {improvement(rmse_p_filt, rmse_p_smth):>14.2f}%")
#     print(f"{'Vel RMSE (m/s)':<15} | {rmse_v_filt:<15.4f} | {rmse_v_smth:<15.4f} | {improvement(rmse_v_filt, rmse_v_smth):>14.2f}%")
#     print(f"{'AngVel RMSE (rad/s)':<15} | {rmse_w_filt:<15.4f} | {rmse_w_smth:<15.4f} | {improvement(rmse_w_filt, rmse_w_smth):>14.2f}%")
#     print("="*50 + "\n")

