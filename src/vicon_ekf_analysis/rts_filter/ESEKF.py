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
        self.x.q = (Rotation.from_rotvec(self.x.w * dt) * self.x.rot()).as_quat() #TODO: Original
        # self.x.q = (self.x.rot() * Rotation.from_rotvec(self.x.w * dt)).as_quat()
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
    timestamps = [None] * n_steps
    timestamps[-1] = history[-1].t
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
        timestamps[k] = history[k].t

    return smoothed_states, timestamps


def fd_body_omega(quats: np.ndarray, times: np.ndarray) -> np.ndarray:
    """
    quats: Nx4 [x,y,z,w] world<-body
    returns ω_body at each sample (central diff; endpoints copied)
    """
    N = len(times)
    Rlist = Rotation.from_quat(quats)
    w_body = np.zeros((N,3))
    for k in range(1, N-1):
        dt = times[k+1] - times[k-1]
        # world (spatial) ang vel via central log
        dq = Rlist[k-1].inv() * Rlist[k+1]
        dtheta = dq.as_rotvec()
        w_world = dtheta / dt
        # convert to body at time k (mid): ω_b = R^T ω_world
        Rk = Rlist[k].as_matrix()
        w_body[k,:] = Rk.T @ w_world
    w_body[0,:]  = w_body[1,:]
    w_body[-1,:] = w_body[-2,:]
    return w_body

# -------------------- Synthetic scenario + test --------------------
def simulate_truth(T=10.0, dt=0.01):
    t = np.arange(0.0, T+dt, dt)
    N = len(t)
    v_world = np.array([0.5, -0.2, 0.1])  # m/s
    w_body  = np.array([0.25, -0.10, 0.40])  # rad/s
    p = np.cumsum(np.vstack([np.zeros(3), np.tile(v_world*dt, (N-1,1))]), axis=0)
    q = np.zeros((N,4))
    R = Rotation.identity()
    q[0,:] = R.as_quat()
    for k in range(1, N):
        R = R * Rotation.from_rotvec(w_body*dt)
        q[k,:] = R.as_quat()
    return t, p, np.tile(v_world,(N,1)), q, np.tile(w_body,(N,1))

def make_measurements(t, p_true, q_true, sigma_p=0.002, sigma_theta=0.003, seed=0):
    rng = np.random.default_rng(seed)
    N = len(t)
    p_meas = p_true + rng.normal(0, sigma_p, size=p_true.shape)
    q_meas = np.zeros_like(q_true)
    for k in range(N):
        dth = rng.normal(0, sigma_theta, size=3)
        Rm  = Rotation.from_rotvec(dth) * Rotation.from_quat(q_true[k])
        q_meas[k,:] = Rm.as_quat()
    return p_meas, q_meas

def run_filter_and_smoother(t, p_meas, q_meas, noise=NoiseParams()):
    ekf = ViconESEKFConstOmega(noise=noise)
    # initialize near truth (zero)
    hist = []
    N = len(t)
    # seed small prior on w to not start at zero exactly
    ekf.x.w = np.array([0.1, -0.05, 0.2])

    # forward pass
    xs_upd = []
    ws_upd = []
    for k in range(N-1):
                
        dt = t[k+1] - t[k]
        x_pred_before = copy_state(ekf.x)
        ekf.predict(dt)
        x_pred_after = copy_state(ekf.x)
        P_pred = ekf.cov.P.copy()
        F      = ekf.F_discrete(dt)

        ekf.update(p_meas[k+1], q_meas[k+1])
        x_upd = copy_state(ekf.x)
        P_upd = ekf.cov.P.copy()

        hist.append(FilterHistoryPoint(
            t=t[k+1], dt=dt, F=F, x_pred=x_pred_after, P_pred=P_pred,
            x_upd=x_upd, P_upd=P_upd
        ))
        xs_upd.append(x_upd)
        ws_upd.append(x_upd.w.copy())

    xs_upd = xs_upd
    ws_upd = np.vstack(ws_upd)
    sm_states, ts = rts_smooth(hist)
    qs_sm = np.vstack([Rotation.from_quat(s.q).as_quat() for s in sm_states])
    ws_sm = np.vstack([s.w for s in sm_states])
    ps_sm = np.vstack([s.p for s in sm_states])
    return np.array(ts), ps_sm, qs_sm, ws_sm, ws_upd


def rmse(a, b):  # rows = time, cols = dims
    d = a - b
    return np.sqrt(np.mean(d**2, axis=0))

def main():
    T, dt = 12.0, 0.01
    t, p_true, v_true, q_true, w_true = simulate_truth(T, dt)
    p_meas, q_meas = make_measurements(t, p_true, q_true)

    ts, p_s, q_s, w_s, w_ekf = run_filter_and_smoother(t, p_meas, q_meas)

    # align shapes (history excludes t[0])
    idx = np.arange(1, len(t))
    t_h   = t[idx]
    w_gt  = w_true[idx]
    q_sm  = q_s
    w_fd  = fd_body_omega(q_sm, t_h)

    # Metrics
    print("=== RMSE (rad/s) ===")
    print("EKF w vs GT:        ", rmse(w_ekf, w_gt))
    print("RTS w vs GT:        ", rmse(w_s,   w_gt))
    print("FD(q_s) vs GT:      ", rmse(w_fd,  w_gt))
    print("RTS w vs FD(q_s):   ", rmse(w_s,   w_fd))

    # Quick sanity on pose/orientation fit (should be tiny)
    pos_err = np.linalg.norm(p_s - p_true[idx], axis=1).mean()
    print("\nMean |pos error| after RTS (m):", pos_err)

if __name__ == "__main__":
    # main()
    
    def which_frame_is_w(q, w, dt=1e-3):
        """
        q: quat [x,y,z,w] for R (world<-body)
        w: your state 'w' (whatever frame it really is)
        """
        Rk = Rotation.from_quat(q)
        # your propagation (the one that works for you):
        R_next = Rotation.from_rotvec(w*dt) * Rk   # <- your current line

        # hypothesis A: w is SPATIAL/world
        R_next_A = Rotation.from_rotvec(w*dt) * Rk

        # hypothesis B: w is BODY
        R_next_B = Rk * Rotation.from_rotvec(w*dt)

        # compare (Frobenius norm)
        A_err = np.linalg.norm((R_next_A.as_matrix() - R_next.as_matrix()))
        B_err = np.linalg.norm((R_next_B.as_matrix() - R_next.as_matrix()))
        return A_err, B_err

    # usage:
    q0 = Rotation.identity().as_quat()
    w  = np.array([0.3, -0.2, 0.4])  # whatever you use
    A_err, B_err = which_frame_is_w(q0, w)
    print("Spatial match error:", A_err)
    print("Body    match error:", B_err)