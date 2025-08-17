"""
Test script for translation-only rigid body EKF with synthetic trajectory data.
Generates known trajectories, adds noise, filters, and validates performance.
"""

import numpy as np
import plotly.graph_objects as go
import plotly.subplots as sp
from plotly.colors import qualitative
from dataclasses import dataclass
from typing import List, Tuple, Optional
import time

# Import our implementations (adjust paths as needed)
from rts_filter import RTS_CKF, ModelFns  # Your EKF implementation
from translation_model_only import (
    RigidBodyState, 
    create_rigid_body_model, 
    create_example_rigid_body_filter
)

def generate_circular_trajectory(
    duration: float = 10.0,
    dt: float = 0.01,
    radius: float = 2.0,
    height: float = 1.0,
    angular_freq: float = 0.5
) -> Tuple[np.ndarray, List[RigidBodyState], List[np.ndarray]]:
    """
    Generate a circular trajectory (translation only).
    
    Returns:
        t: time vector
        true_states: list of true RigidBodyState objects
        measurements: list of measurement vectors [pos(3)]
    """
    
    t = np.arange(0, duration, dt)
    T = len(t)
    
    true_states = []
    measurements = []
    
    for i, time_i in enumerate(t):
        # Circular position trajectory
        theta = angular_freq * time_i
        pos = np.array([
            radius * np.cos(theta),
            radius * np.sin(theta), 
            height
        ])
        
        # Velocity (tangential to circle)
        vel = np.array([
            -radius * angular_freq * np.sin(theta),
            radius * angular_freq * np.cos(theta),
            0.0
        ])
        
        # Create true state (translation only)
        true_state = RigidBodyState(
            position=pos,
            velocity=vel
        )
        true_states.append(true_state)
        
        # Create perfect measurement (position only)
        measurement = pos.copy()
        measurements.append(measurement)
    
    return t, true_states, measurements

def add_measurement_noise(
    measurements: List[np.ndarray],
    pos_noise_std: float = 0.005,
    dropout_rate: float = 0.05,
    seed: int = 42
) -> List[Optional[np.ndarray]]:
    """Add realistic noise to position measurements with occasional dropouts."""
    
    np.random.seed(seed)
    noisy_measurements = []
    
    for meas in measurements:
        # Random dropout
        if np.random.random() < dropout_rate:
            noisy_measurements.append(None)
            continue
            
        # Add position noise
        pos_noisy = meas + np.random.normal(0, pos_noise_std, 3)
        noisy_measurements.append(pos_noisy)
    
    return noisy_measurements

def compute_errors(
    true_states: List[RigidBodyState],
    estimated_nominals: List[RigidBodyState],
    estimated_errors: np.ndarray
) -> dict:
    """Compute various error metrics between true and estimated trajectories."""
    
    T = len(true_states)
    
    pos_errors = np.zeros(T)
    vel_errors = np.zeros(T)
    
    for i in range(T):
        true_state = true_states[i]
        est_nominal = estimated_nominals[i]
        est_error = estimated_errors[i]
        
        # Compose full estimated state
        est_full_pos = est_nominal.position + est_error[:3]
        est_full_vel = est_nominal.velocity + est_error[3:6]
        
        # Position error (Euclidean distance)
        pos_errors[i] = np.linalg.norm(true_state.position - est_full_pos)
        
        # Velocity error
        vel_errors[i] = np.linalg.norm(true_state.velocity - est_full_vel)
    
    return {
        'position': pos_errors,
        'velocity': vel_errors,
        'position_rmse': np.sqrt(np.mean(pos_errors**2)),
        'velocity_rmse': np.sqrt(np.mean(vel_errors**2))
    }

def plot_results(
    t: np.ndarray,
    true_states: List[RigidBodyState],
    filter_results,
    smooth_results: Optional[Tuple[np.ndarray, np.ndarray]] = None,
    errors_filt: dict = None,
    errors_smooth: dict = None
):
    """Plot trajectory comparison and error analysis using Plotly."""
    
    # Extract true trajectory
    true_pos = np.array([s.position for s in true_states])
    true_vel = np.array([s.velocity for s in true_states])
    
    # Extract filtered estimates
    filt_pos = np.array([s.position for s in filter_results.nom_filt]) + filter_results.x_f[:, :3]
    filt_vel = np.array([s.velocity for s in filter_results.nom_filt]) + filter_results.x_f[:, 3:6]
    
    # Extract smoothed estimates if available
    if smooth_results is not None:
        x_s, P_s = smooth_results
        smooth_pos = np.array([s.position for s in filter_results.nom_filt]) + x_s[:, :3]
        smooth_vel = np.array([s.velocity for s in filter_results.nom_filt]) + x_s[:, 3:6]
    
    # Create subplots
    fig = sp.make_subplots(
        rows=3, cols=2,
        subplot_titles=['XY Trajectory', '3D Trajectory', 'Position Error', 
                       'Velocity Error', 'Speed Comparison', 'Covariance Trace'],
        specs=[[{"type": "scatter"}, {"type": "scatter3d"}],
               [{"type": "scatter"}, {"type": "scatter"}],
               [{"type": "scatter"}, {"type": "scatter"}]],
        vertical_spacing=0.08,
        horizontal_spacing=0.10
    )
    
    colors = qualitative.Set1
    
    # 1. XY Trajectory plot
    fig.add_trace(
        go.Scatter(x=true_pos[:, 0], y=true_pos[:, 1], 
                  mode='lines', name='True', 
                  line=dict(color=colors[2], width=3),
                  showlegend=True),
        row=1, col=1
    )
    fig.add_trace(
        go.Scatter(x=filt_pos[:, 0], y=filt_pos[:, 1],
                  mode='lines', name='Filtered',
                  line=dict(color=colors[0], width=2, dash='dash'),
                  showlegend=True),
        row=1, col=1
    )
    if smooth_results is not None:
        fig.add_trace(
            go.Scatter(x=smooth_pos[:, 0], y=smooth_pos[:, 1],
                      mode='lines', name='Smoothed',
                      line=dict(color=colors[1], width=2, dash='dot'),
                      showlegend=True),
            row=1, col=1
        )
    
    fig.update_xaxes(title_text="X (m)", row=1, col=1)
    fig.update_yaxes(title_text="Y (m)", row=1, col=1, scaleanchor="x1", scaleratio=1)
    
    # 2. 3D Trajectory plot
    fig.add_trace(
        go.Scatter3d(x=true_pos[:, 0], y=true_pos[:, 1], z=true_pos[:, 2],
                    mode='lines', name='True 3D',
                    line=dict(color=colors[2], width=6),
                    showlegend=False),
        row=1, col=2
    )
    fig.add_trace(
        go.Scatter3d(x=filt_pos[:, 0], y=filt_pos[:, 1], z=filt_pos[:, 2],
                    mode='lines', name='Filtered 3D',
                    line=dict(color=colors[0], width=4),
                    showlegend=False),
        row=1, col=2
    )
    if smooth_results is not None:
        fig.add_trace(
            go.Scatter3d(x=smooth_pos[:, 0], y=smooth_pos[:, 1], z=smooth_pos[:, 2],
                        mode='lines', name='Smoothed 3D',
                        line=dict(color=colors[1], width=4),
                        showlegend=False),
            row=1, col=2
        )
    
    # 3. Position errors over time
    if errors_filt is not None:
        fig.add_trace(
            go.Scatter(x=t, y=errors_filt['position'],
                      mode='lines', name='Filtered Error',
                      line=dict(color=colors[0], width=2),
                      showlegend=False),
            row=2, col=1
        )
    if errors_smooth is not None:
        fig.add_trace(
            go.Scatter(x=t, y=errors_smooth['position'],
                      mode='lines', name='Smoothed Error',
                      line=dict(color=colors[1], width=2),
                      showlegend=False),
            row=2, col=1
        )
    
    fig.update_xaxes(title_text="Time (s)", row=2, col=1)
    fig.update_yaxes(title_text="Position Error (m)", row=2, col=1)
    
    # 4. Velocity errors
    if errors_filt is not None:
        fig.add_trace(
            go.Scatter(x=t, y=errors_filt['velocity'],
                      mode='lines', name='Filtered Vel Error',
                      line=dict(color=colors[0], width=2),
                      showlegend=False),
            row=2, col=2
        )
    if errors_smooth is not None:
        fig.add_trace(
            go.Scatter(x=t, y=errors_smooth['velocity'],
                      mode='lines', name='Smoothed Vel Error',
                      line=dict(color=colors[1], width=2),
                      showlegend=False),
            row=2, col=2
        )
    
    fig.update_xaxes(title_text="Time (s)", row=2, col=2)
    fig.update_yaxes(title_text="Velocity Error (m/s)", row=2, col=2)
    
    # 5. Speed comparison
    fig.add_trace(
        go.Scatter(x=t, y=np.linalg.norm(true_vel, axis=1),
                  mode='lines', name='True Speed',
                  line=dict(color=colors[2], width=3),
                  showlegend=False),
        row=3, col=1
    )
    fig.add_trace(
        go.Scatter(x=t, y=np.linalg.norm(filt_vel, axis=1),
                  mode='lines', name='Filtered Speed',
                  line=dict(color=colors[0], width=2, dash='dash'),
                  showlegend=False),
        row=3, col=1
    )
    if smooth_results is not None:
        fig.add_trace(
            go.Scatter(x=t, y=np.linalg.norm(smooth_vel, axis=1),
                      mode='lines', name='Smoothed Speed',
                      line=dict(color=colors[1], width=2, dash='dot'),
                      showlegend=False),
            row=3, col=1
        )
    
    fig.update_xaxes(title_text="Time (s)", row=3, col=1)
    fig.update_yaxes(title_text="Speed (m/s)", row=3, col=1)
    
    # 6. Covariance trace (uncertainty)
    cov_trace_filt = np.array([np.trace(P) for P in filter_results.P_f])
    fig.add_trace(
        go.Scatter(x=t, y=cov_trace_filt,
                  mode='lines', name='Filtered Uncertainty',
                  line=dict(color=colors[0], width=2),
                  showlegend=False),
        row=3, col=2
    )
    if smooth_results is not None:
        cov_trace_smooth = np.array([np.trace(P) for P in P_s])
        fig.add_trace(
            go.Scatter(x=t, y=cov_trace_smooth,
                      mode='lines', name='Smoothed Uncertainty',
                      line=dict(color=colors[1], width=2),
                      showlegend=False),
            row=3, col=2
        )
    
    fig.update_xaxes(title_text="Time (s)", row=3, col=2)
    fig.update_yaxes(title_text="Trace(P)", type="log", row=3, col=2)
    
    # Update layout
    fig.update_layout(
        title_text="Translation-Only Rigid Body EKF Performance Analysis",
        title_x=0.5,
        height=900,
        width=1200,
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right", 
            x=1
        )
    )
    
    # Add grid to 2D subplots
    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='LightPink')
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='LightPink')
    
    return fig

def run_test():
    """Main test function."""
    
    print("=== Translation-Only Rigid Body EKF Test ===\n")
    
    # Generate synthetic trajectory
    print("1. Generating synthetic trajectory...")
    t, true_states, perfect_measurements = generate_circular_trajectory(
        duration=20.0,
        dt=0.02,  # 50 Hz
        radius=3.0,
        angular_freq=0.3
    )
    print(f"   Generated {len(t)} timesteps over {t[-1]:.1f} seconds")
    
    # Add noise to measurements
    print("\n2. Adding measurement noise and dropouts...")
    noisy_measurements = add_measurement_noise(
        perfect_measurements,
        pos_noise_std=0.003,    # 3mm position noise
        dropout_rate=0.00       # 2% measurement dropout
    )
    
    num_dropouts = sum(1 for m in noisy_measurements if m is None)
    print(f"   Added noise, {num_dropouts} measurement dropouts ({100*num_dropouts/len(t):.1f}%)")
    
    # Create filter
    print("\n3. Setting up EKF...")
    model, n_states, x0, P0, nom0 = create_example_rigid_body_filter()
    ekf = RTS_CKF(n_states, model, jitter=1e-10)
    print(f"   Filter initialized with {n_states} error states")
    
    # Run filter
    print("\n4. Running EKF forward pass...")
    start_time = time.time()
    filter_results = ekf.forward(t, noisy_measurements, x0, P0, nom0)
    filter_time = time.time() - start_time
    print(f"   Forward pass completed in {filter_time:.3f} seconds ({len(t)/filter_time:.0f} Hz)")
    
    # Run smoother
    print("\n5. Running RTS smoother...")
    start_time = time.time()
    smooth_results = ekf.smooth(filter_results)
    smooth_time = time.time() - start_time
    print(f"   Smoothing completed in {smooth_time:.3f} seconds")
    
    # Compute errors
    print("\n6. Computing estimation errors...")
    errors_filt = compute_errors(true_states, filter_results.nom_filt, filter_results.x_f)
    errors_smooth = compute_errors(true_states, filter_results.nom_filt, smooth_results[0])

    print("CLOSOSOSOS --- ", np.max(errors_filt['position'] - errors_smooth['position']))
    print("CLOSOSOSOS --- ", np.max(errors_filt['velocity'] - errors_smooth['velocity']))
    
    # Print results
    print("\n=== RESULTS ===")
    print(f"Filter RMSE:")
    print(f"  Position:     {errors_filt['position_rmse']*1000:.2f} mm")
    print(f"  Velocity:     {errors_filt['velocity_rmse']*1000:.2f} mm/s")
    
    print(f"\nSmoother RMSE:")
    print(f"  Position:     {errors_smooth['position_rmse']*1000:.2f} mm") 
    print(f"  Velocity:     {errors_smooth['velocity_rmse']*1000:.2f} mm/s")
    
    improvement = {
        'position': (errors_filt['position_rmse'] - errors_smooth['position_rmse']) / errors_filt['position_rmse'] * 100,
        'velocity': (errors_filt['velocity_rmse'] - errors_smooth['velocity_rmse']) / errors_filt['velocity_rmse'] * 100
    }
    
    print(f"\nSmoother improvement over filter:")
    print(f"  Position:     {improvement['position']:.1f}%")
    print(f"  Velocity:     {improvement['velocity']:.1f}%")
    
    # Validation checks
    print("\n=== VALIDATION ===")
    passed = True
    
    # Check if errors are reasonable for translation-only model
    pos_threshold = 0.01  # 1cm
    vel_threshold = 0.05  # 5cm/s
    
    if errors_smooth['position_rmse'] > pos_threshold:
        print(f"❌ Position RMSE too high: {errors_smooth['position_rmse']*1000:.1f} mm > {pos_threshold*1000:.1f} mm")
        passed = False
    else:
        print(f"✅ Position RMSE acceptable: {errors_smooth['position_rmse']*1000:.1f} mm")
        
    if errors_smooth['velocity_rmse'] > vel_threshold:
        print(f"❌ Velocity RMSE too high: {errors_smooth['velocity_rmse']*1000:.1f} mm/s > {vel_threshold*1000:.1f} mm/s")
        passed = False
    else:
        print(f"✅ Velocity RMSE acceptable: {errors_smooth['velocity_rmse']*1000:.1f} mm/s")
    
    # Check smoother improvement
    if improvement['position'] < 5:
        print(f"⚠️  Small position improvement from smoothing: {improvement['position']:.1f}%")
    else:
        print(f"✅ Good position improvement from smoothing: {improvement['position']:.1f}%")
        
    if improvement['velocity'] < 5:
        print(f"⚠️  Small velocity improvement from smoothing: {improvement['velocity']:.1f}%")
    else:
        print(f"✅ Good velocity improvement from smoothing: {improvement['velocity']:.1f}%")
    
    print(f"\n{'✅ ALL TESTS PASSED!' if passed else '❌ SOME TESTS FAILED'}")
    
    # Plot results
    print("\n7. Generating interactive plots...")
    plot_fig = plot_results(t, true_states, filter_results, smooth_results, errors_filt, errors_smooth)
    
    return passed, filter_results, smooth_results, errors_filt, errors_smooth, plot_fig

if __name__ == "__main__":
    # Run the test
    try:
        success, filt_res, smooth_res, err_f, err_s, fig = run_test()
        print(f"\nTest completed {'successfully' if success else 'with issues'}!")
        
        # Show plot if successful
        if success:
            print("\n8. Displaying results...")
            fig.write_html("translation_only_rigid_body_ekf_results.html")
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()