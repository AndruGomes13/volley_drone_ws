import numpy as np
from dataclasses import dataclass
from typing import Any, Callable, Optional, Tuple
from scipy.linalg import cho_factor, cho_solve

# ---------- Core callback interface ----------

@dataclass
class ModelFns:
    """
    Generic manifold-aware EKF/RTS interface.
    All math is in error-space R^n; 'nom' is your nominal state on any manifold.
    Provide consistent linearizations and retraction/residual ops.
    """

    # Time propagation of the *nominal* state (may use parts of x_err and controls u).
    propagate_nominal: Callable[[Any, np.ndarray, float, Optional[Any]], Any]
    # Discrete-time linearization for the error-state: x_{k+1|k} = A_k x_k + w,  w~N(0,Q_k)
    A: Callable[[int, float, Any, np.ndarray, Optional[Any]], np.ndarray]   # (n,n)
    Q: Callable[[int, float, Any, np.ndarray, Optional[Any]], np.ndarray]   # (n,n)

    # Measurement model at time k (linearized about current nominal):
    predict_meas: Callable[[int, Any], np.ndarray]                          # y_hat in R^m
    H: Callable[[int, Any], np.ndarray]                                     # (m,n)
    R: Callable[[int], np.ndarray]                                          # (m,m)

    # Geometry: residual on measurement space, and retraction on state manifold.
    residual_meas: Callable[[np.ndarray, np.ndarray], np.ndarray]           # (m,)
    compose: Callable[[Any, np.ndarray], Any]                               # nom' = compose(nom, δx)

    # Error-state reset after update (set some components to 0, shrink P blocks, etc.)
    reset_error: Callable[[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]

# ---------- Filter + Smoother ----------

@dataclass
class KFRTSOutputs:
    # Stored for RTS
    A_list: list
    x_p: np.ndarray   # (T,n)  predicted means (prior to update at k)
    P_p: np.ndarray   # (T,n,n)
    x_f: np.ndarray   # (T,n)  filtered means (posterior at k)
    P_f: np.ndarray   # (T,n,n)
    nom_pred: list    # predicted nominals (before update) length T
    nom_filt: list    # filtered nominals (after update)  length T

class RTS_CKF:
    """
    Generic error-state EKF with RTS smoothing.
    Works for Euclidean and manifold states via user-supplied callbacks.
    """

    def __init__(self, n: int, model: ModelFns, auto_reset: bool = True, jitter: float = 1e-12):
        self.n = n
        self.auto_reset = auto_reset
        self.mf = model
        self.jitter = jitter

    def forward(
        self,
        t: np.ndarray,                    # shape (T,)
        y: list,                          # list/array of measurements (len T), allow None
        x0: np.ndarray,                   # shape (n,)   initial error-state (usually zeros)
        P0: np.ndarray,                   # shape (n,n)  initial covariance
        nom0: Any,                        # initial nominal (on manifold or Euclidean)
        u: Optional[list] = None          # optional controls (len T or len T-1), passed to callbacks
    ) -> KFRTSOutputs:
        T = len(t); assert T == len(y)
        if u is not None:
            assert len(u) in (T, T-1)

        # Storage
        x_p = np.zeros((T, self.n))
        P_p = np.zeros((T, self.n, self.n))
        x_f = np.zeros((T, self.n))
        P_f = np.zeros((T, self.n, self.n))
        A_list = [np.eye(self.n)] * T
        nom_pred = [None] * T
        nom_filt = [None] * T

        # Init (k=0 prior is the given x0,P0, nominal = nom0)
        xk = x0.copy()
        Pk = P0.copy()
        nom_k = nom0

        # We allow an update at k=0 using the *current* nominal (no time propagation yet)
        A0 = np.eye(self.n)
        A_list[0] = A0
        x_p[0], P_p[0] = xk.copy(), Pk.copy()
        nom_pred[0] = nom_k

        # --- Update at k=0 (if measurement present) ---
        if y[0] is not None:
            yhat = self.mf.predict_meas(0, nom_pred[0])
            zk = self.mf.residual_meas(y[0], yhat)             # innovation in R^m
            Hk = self.mf.H(0, nom_pred[0])
            Rk = self._sym(self.mf.R(0))
            S = self._sym(Hk @ Pk @ Hk.T + Rk); S += np.eye(S.shape[0]) * self.jitter
            cS = cho_factor(S, lower=True, check_finite=False)
            # K = cho_solve(cS, (Hk @ Pk).T, check_finite=False).T   # (n,m)
            K = cho_solve(cS, Hk @ Pk, check_finite=False).T   # (n,m)

            innov = zk - Hk @ xk
            xk = xk + K @ innov
            Pk = self._joseph(Pk, K, Hk, Rk)
            # Apply error to nominal & reset
            nom_k = self.mf.compose(nom_pred[0], xk)
            xk, Pk = self.mf.reset_error(xk, Pk)

        x_f[0], P_f[0] = xk.copy(), Pk.copy()
        nom_filt[0] = nom_k

        # --- Main loop k = 1..T-1 ---
        for k in range(1, T):
            dt = float(t[k] - t[k-1]);  assert dt > 0.0
            ukm1 = None if u is None else (u[k-1] if len(u) == T-1 else u[k])

            # Linearization and process noise for error-state
            Ak = self.mf.A(k-1, dt, nom_k, xk, ukm1)
            Qk = self._sym(self.mf.Q(k-1, dt, nom_k, xk, ukm1))

            # Predict error-state
            xk_pred = Ak @ xk
            Pk_pred = Ak @ Pk @ Ak.T + Qk

            # Predict nominal state (user-defined manifold dynamics)
            nom_pred_k = self.mf.propagate_nominal(nom_k, xk, dt, ukm1)

            # Store priors for this step (before using y[k])
            x_p[k], P_p[k] = xk_pred.copy(), Pk_pred.copy()
            nom_pred[k] = nom_pred_k
            A_list[k-1] = Ak

            # If measurement is available, update
            if y[k] is not None:
                yhat = self.mf.predict_meas(k, nom_pred_k)
                zk = self.mf.residual_meas(y[k], yhat)
                Hk = self.mf.H(k, nom_pred_k)
                Rk = self._sym(self.mf.R(k))

                S = self._sym(Hk @ Pk_pred @ Hk.T + Rk); S += np.eye(S.shape[0]) * self.jitter
                cS = cho_factor(S, lower=True, check_finite=False)
                # K = cho_solve(cS, (Hk @ Pk_pred).T, check_finite=False).T   # (n,m)
                K = cho_solve(cS, Hk @ Pk_pred, check_finite=False).T   # (n,m)

                innov = zk - Hk @ xk_pred
                xk = xk_pred + K @ innov
                Pk = self._joseph(Pk_pred, K, Hk, Rk)

                # Apply error to nominal & reset
                if self.auto_reset:
                    nom_k = self.mf.compose(nom_pred_k, xk)
                    xk, Pk = self.mf.reset_error(xk, Pk)
                else:
                    nom_k = self.mf.compose(nom_pred_k, xk)

            else:
                # No measurement: carry priors
                xk, Pk = xk_pred, Pk_pred
                nom_k = nom_pred_k

            x_f[k], P_f[k] = xk.copy(), Pk.copy()
            nom_filt[k] = nom_k

        return KFRTSOutputs(A_list=A_list, x_p=x_p, P_p=P_p, x_f=x_f, P_f=P_f,
                            nom_pred=nom_pred, nom_filt=nom_filt)

    def smooth(self, outs: KFRTSOutputs) -> Tuple[np.ndarray, np.ndarray]:
        """Standard linear RTS using stored A, P_f, P_p."""
        T, n = outs.x_f.shape[0], outs.x_f.shape[1]
        x_s = outs.x_f.copy()
        P_s = outs.P_f.copy()

        for k in range(T-2, -1, -1):
            Ak = outs.A_list[k]                        # transition from k -> k+1
            Ppk1 = outs.P_p[k+1]                       # prior at k+1
            L, low = cho_factor(self._sym(Ppk1), lower=True, check_finite=False)
            rhs = (outs.P_f[k] @ Ak.T).T                 # (n, n)
            Ck  = cho_solve((L, low), rhs, check_finite=False).T  # (n, n)
            dx = x_s[k+1] - outs.x_p[k+1]
            x_s[k] = outs.x_f[k] + Ck @ dx
            P_s[k] = outs.P_f[k] + Ck @ (P_s[k+1] - Ppk1) @ Ck.T
        return x_s, P_s

    # ----- helpers -----
    @staticmethod
    def _sym(M: np.ndarray) -> np.ndarray:
        return 0.5 * (M + M.T)

    @staticmethod
    def _joseph(P_pred: np.ndarray, K: np.ndarray, H: np.ndarray, R: np.ndarray) -> np.ndarray:
        I = np.eye(P_pred.shape[0])
        return (I - K @ H) @ P_pred @ (I - K @ H).T + K @ R @ K.T
