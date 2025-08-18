from dataclasses import dataclass
from scipy.spatial.transform import Rotation as R
from scipy.spatial.transform import Slerp
import numpy as np
from scipy import signal
import scipy.interpolate


# --- Classes ---
@dataclass
class XYZAnalysis:
    mean: np.ndarray
    std: np.ndarray
    percentile_95: np.ndarray
    percentile_99: np.ndarray
    max_deviation: np.ndarray
    rms: np.ndarray
    covariance: np.ndarray
    
@dataclass
class OrientationAnalysis:
    mean: R
    std: np.ndarray
    vec_rms: np.ndarray
    vec_max_deviation: np.ndarray
    vec_covariance: np.ndarray

@dataclass
class ACFAnalysis:
    lags: np.ndarray
    acf_values: np.ndarray
    conf_band: np.ndarray
    
# --- Printing Fucntions ---
def print_xyz_analysis(analysis: XYZAnalysis, label: str = "Position"):
    # --- Mean ---
    print(f"{label} Mean:\n X: {analysis.mean[0]:.4f};    Y: {analysis.mean[1]:.4f};    Z: {analysis.mean[2]:.4f}")

    # --- Standard Deviation ---
    print(f"{label} Std Dev:\n X: {analysis.std[0]:.8f};    Y: {analysis.std[1]:.8f};    Z: {analysis.std[2]:.8f}")

    # --- Percentiles ---
    print(f"{label} 95th Percentile:\n X: {analysis.percentile_95[0]:.8f};    Y: {analysis.percentile_95[1]:.8f};    Z: {analysis.percentile_95[2]:.8f}")
    print(f"{label} 99th Percentile:\n X: {analysis.percentile_99[0]:.8f};    Y: {analysis.percentile_99[1]:.8f};    Z: {analysis.percentile_99[2]:.8f}")

    # --- Max Deviation ---
    print(f"{label} Max Deviation:\n X: {analysis.max_deviation[0]:.8f};    Y: {analysis.max_deviation[1]:.8f};    Z: {analysis.max_deviation[2]:.8f}")
    
    # --- RMS ---
    print(f"{label} RMS:\n X: {analysis.rms[0]:.8f};    Y: {analysis.rms[1]:.8f};    Z: {analysis.rms[2]:.8f}")

    # --- Covariance Analysis ---
    print(f"{label} Covariance Matrix (rows/cols: X, Y, Z):")
    print("         X           Y           Z")
    for i, axis in enumerate(['X', 'Y', 'Z']):
        print(f"{axis}  " + "  ".join(f"{analysis.covariance[i, j]:.4e}" for j in range(3)))

def print_orientation_analysis(analysis: OrientationAnalysis, label: str = "Orientation"):
    # --- Mean ---
    mean_quat = analysis.mean.as_quat(scalar_first=True)
    print(f"{label} Mean (as quaternion):\n W: {mean_quat[0]:.5f};    X: {mean_quat[1]:.5f};    Y: {mean_quat[2]:.5f};    Z: {mean_quat[3]:.5f}")

    # --- Standard Deviation ---
    std_rotvec = analysis.std
    print(f"{label} Std Dev (as deg):\n X: {np.rad2deg(std_rotvec[0]):.8f};    Y: {np.rad2deg(std_rotvec[1]):.8f};    Z: {np.rad2deg(std_rotvec[2]):.8f}")

    # --- Vector RMS ---
    vec_rms = analysis.vec_rms
    print(f"{label} Vector RMS (as deg):\n X: {np.rad2deg(vec_rms[0]):.8f};    Y: {np.rad2deg(vec_rms[1]):.8f};    Z: {np.rad2deg(vec_rms[2]):.8f}")
    
    # --- Vector Max Deviation ---
    vec_max_deviation = analysis.vec_max_deviation
    print(f"{label} Vector Max Deviation (as deg):\n X: {np.rad2deg(vec_max_deviation[0]):.8f};    Y: {np.rad2deg(vec_max_deviation[1]):.8f};    Z: {np.rad2deg(vec_max_deviation[2]):.8f}")

    # --- Covariance Analysis ---
    print(f"{label} Covariance Matrix (rows/cols: x, y, z):")
    print("         x           y           z")
    for i, axis in enumerate(['x', 'y', 'z']):
        print(f"{axis}  " + "  ".join(f"{analysis.vec_covariance[i, j]:.4e}" for j in range(3)))

def print_acf_analysis(analysis: ACFAnalysis, top_k: int = 10):
    """
    Print a summary of ACF results:
    - 95% confidence interval
    - number of significant lags
    - largest correlations (by magnitude)
    """
    lags = analysis.lags
    acf = analysis.acf_values
    ci_low, ci_high = analysis.conf_band
    sig_mask = (acf < ci_low) | (acf > ci_high)
    sig_lags = lags[sig_mask]
    sig_vals = acf[sig_mask]

    print("=== ACF Analysis Summary ===")
    print(f"95% confidence band under white-noise null: [{ci_low:.4f}, {ci_high:.4f}]")
    print(f"Total lags checked: {len(lags)-1} (excluding lag=0)")
    print(f"Significant lags: {len(sig_lags)}")

    if len(sig_lags) > 0:
        # Show top_k strongest correlations
        order = np.argsort(-np.abs(sig_vals))
        print(f"\nTop {min(top_k, len(sig_lags))} significant lags:")
        for idx in order[:top_k]:
            print(f"  Lag {sig_lags[idx]:3d}: ACF={sig_vals[idx]:+.4f}")
    else:
        print("No significant correlations detected — consistent with white noise.")

# --- Analysis Functions ---
def mean_rotation(Rs: R, max_iters=50, tol=1e-10) -> R:
    """
    Intrinsic (log-exp) mean on SO(3) using scipy Rotations.

    Rs: a Rotation object with batch size N (i.e., R.from_quat(quats) or R.from_matrix(...)).
    Returns: a single Rotation (the mean).
    """
    R_bar = R.identity()
    for _ in range(max_iters):
        # Relative rotations into the tangent at the current estimate
        rel = R_bar.inv() * Rs                          # (N,) Rotation
        delta = rel.as_rotvec().mean(axis=0)            # (3,)
        if np.linalg.norm(delta) < tol:
            break
        R_bar = R_bar * R.from_rotvec(delta)
    return R_bar

def compute_acf(x, max_lag=None, unbiased=True, demean=True):
    """
    Returns lags [0..max_lag], acf values, and ~95% conf. band for white noise.
    """
    x = np.asarray(x)
    if demean:
        x = x - np.mean(x)
    n = x.size
    if max_lag is None:
        max_lag = min(n // 4, 2000)  # sane default

    # full autocorrelation then slice central part
    corr = signal.correlate(x, x, mode="full", method="auto")
    mid = corr.size // 2
    acf = corr[mid:mid + max_lag + 1]

    # normalize by zero-lag
    acf = acf / acf[0]

    if unbiased:
        # unbiased normalization for each lag
        acf = acf * (n / (n - np.arange(0, max_lag + 1)))

    # 95% CI under H0: white noise (large-n)
    # Bartlett’s simple bound: +/- 1.96/sqrt(n) for lags >= 1
    ci = 1.96 / np.sqrt(n)
    return ACFAnalysis(
        lags=np.arange(max_lag + 1),
        acf_values=acf,
        conf_band=np.array([-ci, ci]) if max_lag > 0 else np.array([0, 0])
    )

def xyz_analysis(position_measurements: np.ndarray) -> XYZAnalysis:
    """
    Perform position analysis on the given measurements.
    Returns a dictionary with mean, std, 95th and 99th percentiles, RMS, and covariance.
    """
    position_mean = position_measurements.mean(axis=0)
    position_residuals = position_measurements - position_mean
    
    position_max_deviation = np.abs(position_residuals).max(axis=0)
    
    position_std = position_residuals.std(axis=0)

    position_95th = np.percentile(position_residuals, 95, axis=0)
    position_99th = np.percentile(position_residuals, 99, axis=0)

    position_rms = np.sqrt((position_residuals**2).mean(axis=0))

    position_cov = np.cov(position_residuals, rowvar=False)

    return XYZAnalysis(
        mean=position_mean,
        std=position_std,
        percentile_95=position_95th,
        percentile_99=position_99th,
        max_deviation=position_max_deviation,
        rms=    position_rms,
        covariance=position_cov
    )

def orientation_analysis(orientation_measurements: R) -> OrientationAnalysis:
    """
    Perform orientation analysis on the given measurements.
    Returns a dictionary with mean, std, vector RMS, and covariance.
    """
    orientation_mean = mean_rotation(orientation_measurements)
    orientation_residuals = orientation_measurements * orientation_mean.inv()
    
    orientation_std = orientation_residuals.as_rotvec().std(axis=0)
    
    orientation_vec_rms = np.sqrt((orientation_residuals.as_rotvec()**2).mean(axis=0))
    
    orientation_max_deviation = np.abs(orientation_residuals.as_rotvec()).max(axis=0)
    
    orientation_covariance = np.cov(orientation_residuals.as_rotvec(), rowvar=False)

    return OrientationAnalysis(
        mean=orientation_mean,
        std=orientation_std,
        vec_rms=orientation_vec_rms,
        vec_max_deviation=orientation_max_deviation,
        vec_covariance=orientation_covariance
    )
  
def interpolate_xyz_vector(original_positions: np.ndarray, original_timestamps: np.ndarray, new_timestamps: np.ndarray) -> np.ndarray:
    """
    Interpolate positions to match new timestamps.
    original_positions: Nx3 array of positions (x, y, z).
    original_timestamps: 1D array of timestamps corresponding to original_positions.
    new_timestamps: 1D array of timestamps to interpolate to.
    Returns interpolated positions as a 2D array with shape (len(new_timestamps), 3).
    """
    assert np.all(np.diff(original_timestamps) >= 0), "Original timestamps must be sorted."
    assert np.all(np.diff(new_timestamps) >= 0), "New timestamps must be sorted."
    assert original_positions.shape[1] == 3, "Original positions must have shape Nx3."
    assert original_timestamps.shape[0] == original_positions.shape[0], "Timestamps and positions must match in length."
    assert new_timestamps.ndim == 1, "New timestamps must be a 1D array."
    assert original_timestamps.ndim == 1, "Original timestamps must be a 1D array."
    
    # Validity: queries inside the convex hull of t_src
    valid_mask = (new_timestamps >= original_timestamps[0]) & (new_timestamps <= original_timestamps[-1])
    new_timestamps_valid = new_timestamps[valid_mask]

    if new_timestamps_valid.size == 0:
        raise ValueError("No valid timestamps for interpolation.")
    
    # Interpolate each coordinate separately
    interp_func_x = scipy.interpolate.interp1d(original_timestamps, original_positions[:, 0], kind="quadratic", bounds_error=True)
    interp_func_y = scipy.interpolate.interp1d(original_timestamps, original_positions[:, 1], kind="quadratic", bounds_error=True)
    interp_func_z = scipy.interpolate.interp1d(original_timestamps, original_positions[:, 2], kind="quadratic", bounds_error=True)

    # Stack the interpolated coordinates
    interpolated_positions = np.column_stack((
        interp_func_x(new_timestamps_valid),
        interp_func_y(new_timestamps_valid),
        interp_func_z(new_timestamps_valid)
    ))
    
    return interpolated_positions, valid_mask

def interpolate_orientation(original_orientations: R, original_timestamps: np.ndarray, new_timestamps: np.ndarray) -> R:
    """
    Interpolate orientations to match new timestamps.
    original_orientations: Rotation object with batch size N.
    original_timestamps: 1D array of timestamps corresponding to original_orientations.
    new_timestamps: 1D array of timestamps to interpolate to.
    Returns interpolated orientations as a Rotation object.
    """
    assert np.all(np.diff(original_timestamps) >= 0), "Original timestamps must be sorted."
    assert np.all(np.diff(new_timestamps) >= 0), "New timestamps must be sorted."
    assert original_timestamps.ndim == 1, "Original timestamps must be a 1D array."
    assert new_timestamps.ndim == 1, "New timestamps must be a 1D array."
    
    # Validity: queries inside the convex hull of t_src
    valid_mask = (new_timestamps >= original_timestamps[0]) & (new_timestamps <= original_timestamps[-1])
    new_timestamps_valid = new_timestamps[valid_mask]

    if new_timestamps_valid.size == 0:
        raise ValueError("No valid timestamps for interpolation.")
    
    # Interpolate using spherical linear interpolation (slerp)
    slerp = Slerp(original_timestamps, original_orientations)
    
    return slerp(new_timestamps_valid), valid_mask

# --- Auto regressive noise model ---
def ar1_params(residuals):
    x = residuals - residuals.mean()
    s = x.std(ddof=0)
    r1 = np.corrcoef(x[:-1], x[1:])[0,1]
    phi = r1
    sigma_eps = s * np.sqrt(1 - phi**2)
    return phi, sigma_eps

class AR1Noise:
    def __init__(self, phi, sigma_eps):
        self.phi = float(phi)
        self.sigma = float(sigma_eps)
        self.r = 0.0
    def reset(self, r0=0.0):
        self.r = float(r0)
    def step(self, rng=np.random):
        eps = rng.normal(0.0, self.sigma)
        self.r = self.phi * self.r + eps
        return self.r
    
    
if __name__ == "__main__":
    # Example usage
    print(R.identity())  # Identity rotation
    help(Slerp)