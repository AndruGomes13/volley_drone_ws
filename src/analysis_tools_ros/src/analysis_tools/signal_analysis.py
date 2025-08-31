
from typing import Tuple
import numpy as np
from scipy.signal import correlate, correlation_lags
import scipy.interpolate


# --- Interpolation utilities ---
  
def interpolate_1d_signal(original_signal: np.ndarray, original_timestamps: np.ndarray, new_timestamps: np.ndarray) :
    """
    Interpolate a 1D signal to match new timestamps.
    original_signal: 1D array of signal values.
    original_timestamps: 1D array of timestamps corresponding to original_signal.
    new_timestamps: 1D array of timestamps to interpolate to.
    Returns interpolated signal as a 1D array.
    """
    assert np.all(np.diff(original_timestamps) >= 0), "Original timestamps must be sorted."
    assert np.all(np.diff(new_timestamps) >= 0), "New timestamps must be sorted."
    assert original_signal.ndim == 1, "Original signal must be a 1D array."
    assert original_timestamps.ndim == 1, "Original timestamps must be a 1D array."
    
    # Validity: queries inside the convex hull of t_src
    valid_mask = (new_timestamps >= original_timestamps[0]) & (new_timestamps <= original_timestamps[-1])
    new_timestamps_valid = new_timestamps[valid_mask]

    if new_timestamps_valid.size == 0:
        raise ValueError("No valid timestamps for interpolation.")
    
    interp_signal = scipy.interpolate.interp1d(original_timestamps, original_signal, kind="linear", bounds_error=True)(new_timestamps_valid)

    return interp_signal, valid_mask

  
def interpolate_xyz_vector(original_positions: np.ndarray, original_timestamps: np.ndarray, new_timestamps: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
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
    
    interp_x, valid_mask = interpolate_1d_signal(original_positions[:, 0], original_timestamps, new_timestamps)
    interp_y , _= interpolate_1d_signal(original_positions[:, 1], original_timestamps, new_timestamps)
    interp_z ,_ = interpolate_1d_signal(original_positions[:, 2], original_timestamps, new_timestamps)
    interpolated_positions = np.column_stack((interp_x[0], interp_y[0], interp_z[0]))

    return interpolated_positions, valid_mask



# --- Delay estimation and signal shifting functions ---

def estimate_delay(x, y, dt):
    # zero-mean
    x0 = x - x.mean()
    y0 = y - y.mean()
    c = correlate(y0, x0, mode="full", method="fft")
    lags = correlation_lags(len(y0), len(x0), mode="full")  # samples
    k_idx = int(np.argmax(c))
    lag_samples = int(lags[k_idx])
    tau = lag_samples * dt  # seconds; tau>0 => y lags x
    return lag_samples, tau, c, k_idx

def refine_subsample_peak(c, k_idx, dt):
    # parabolic interpolation around the discrete peak
    if k_idx <= 0 or k_idx >= len(c) - 1:
        return 0.0  # can't refine at boundary
    y1, y2, y3 = c[k_idx - 1], c[k_idx], c[k_idx + 1]
    denom = (y1 - 2*y2 + y3)
    if denom == 0:
        return 0.0
    delta_samples = 0.5 * (y1 - y3) / denom  # in (-0.5, 0.5)
    return float(delta_samples * dt)  # seconds

def shift_signal(t, s, tau):
    # returns s(t - tau); positive tau delays s
    f = scipy.interpolate.interp1d(t, s, kind="linear", fill_value="extrapolate", assume_sorted=True)
    return f(t - tau)

def run_delay_analysis_uniform(signal_x: np.ndarray, signal_y: np.ndarray, dt: float):
    """
    Run delay analysis on two signals.
    Returns the estimated lag, tau, and the aligned signal_y.
    NOTE: Assumed uniform sampling.
    """
    lag_samples, tau, c, k_idx = estimate_delay(signal_x, signal_y, dt)
    delta = refine_subsample_peak(c, k_idx, dt)
    y_aligned = shift_signal(np.arange(len(signal_y)), signal_y, -(tau + delta)) #TODO: check if this is correct, it should be negative because y lags x

    # Print results
    print(f"Estimated lag: {lag_samples} samples, tau: {tau:.5f} s, delta: {delta:.5f} s")
    print(f"Max difference before alignment: {np.rad2deg(np.max(signal_y - signal_x)):.5f} degrees")
    print(f"Max difference after alignment: {np.rad2deg(np.max(y_aligned - signal_x)):.5f} degrees")
    
    return lag_samples, tau, y_aligned

def run_delay_analysis_nonuniform(signal_x: np.ndarray, t_x: np.ndarray, signal_y: np.ndarray, t_y: np.ndarray):
    """
    Run delay analysis on two non-uniformly sampled signals.
    Finds the less sampled signal, uses it's timestamps for finding the average dt, samples uniformly both signals, runs delay analysis.
    """
    assert np.all(np.diff(t_x) >= 0), "Signal X timestamps must be sorted."
    assert np.all(np.diff(t_y) >= 0), "Signal Y timestamps must be sorted."

    t_min = max(t_x[0], t_y[0])
    t_max = min(t_x[-1], t_y[-1])
    
    dt_avg_x = np.mean(np.diff(t_x))
    dt_avg_y = np.mean(np.diff(t_y))
    
    dt_avg = max(dt_avg_x, dt_avg_y)
    
    t_uniform = np.arange(t_min, t_max, dt_avg)
    
    signal_x_uniform, _ = interpolate_1d_signal(signal_x, t_x, t_uniform)
    signal_y_uniform, _ = interpolate_1d_signal(signal_y, t_y, t_uniform)

    lag_samples, tau, c, k_idx = estimate_delay(signal_x_uniform, signal_y_uniform, dt_avg)
    delta = refine_subsample_peak(c, k_idx, dt_avg)
    
    y_aligned = shift_signal(t_y, signal_y, -(tau + delta))  # negative because y lags x

    return lag_samples, tau + delta, y_aligned


