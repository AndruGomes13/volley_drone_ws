from plotly import graph_objects as go
from analysis_utils import ACFAnalysis
import numpy as np
from plotly.subplots import make_subplots


def plot_acf_plotly(analysis: ACFAnalysis, title="ACF with 95% bounds") -> go.Figure:
    ci_low, ci_high = float(analysis.conf_band[0]), float(analysis.conf_band[1])
    fig = go.Figure()

    # Bars for ACF (stem-like look)
    fig.add_trace(go.Bar(
        x=analysis.lags,
        y=analysis.acf_values,
        name="ACF",
        hovertemplate="Lag %{x}<br>ACF %{y:.4f}<extra></extra>",
    ))

    # Shaded 95% band
    fig.add_hrect(y0=ci_low, y1=ci_high, fillcolor="LightGrey", opacity=0.35, line_width=0)

    # Zero line
    fig.add_hline(y=0, line=dict(dash="dash", width=1), opacity=0.7)

    fig.update_layout(
        title=title,
        xaxis_title="Lag",
        yaxis_title="Autocorrelation",
        bargap=0.15,
        template="plotly_white",
        showlegend=False,
    )
    return fig


def add_3d_trajectory_traces(fig: go.Figure, t: np.ndarray, p: np.ndarray,
                          label_prefix="", line=dict()) -> None:
    """Adds 3D position and 2D velocity traces to the given Plotly figure."""
    # 3D Position
    fig.add_trace(go.Scatter3d(
        x=p[:, 0], y=p[:, 1], z=p[:, 2],
        mode='lines',
        name=f"{label_prefix} Position",
        line=line,
        hovertemplate=f"{label_prefix} Position<br>X: %{{x:.2f}}<br>Y: %{{y:.2f}}<br>Z: %{{z:.2f}}<extra></extra>"
    )) 
    return fig

# Utility function to stack 3 velocity component plots vertically using Plotly subplots

def add_xyz_traces_stacked(fig: go.Figure, t: np.ndarray, v: np.ndarray, label_prefix="", line=dict(), row_offset=0):
    """
    Adds 3 velocity component traces (Vx, Vy, Vz) to an existing Plotly Figure with stacked subplots.
    Args:
        fig: Plotly Figure object (should have at least 3 rows).
        t: Time array.
        v: Velocity array (N x 3).
        label_prefix: Prefix for trace names.
        line: Line style dict.
        row_offset: Row offset (0 for rows 1-3, 3 for 4-6, etc).
    Returns:
        The modified figure.
    """
    axes = ['X', 'Y', 'Z']
    for i in range(3):
        fig.add_trace(
            go.Scatter(
                x=t, y=v[:, i],
                mode='lines',
                name=f"{label_prefix} {axes[i]}",
                line=line,
                hovertemplate=f"{label_prefix} {axes[i]}<br>Time: %{{x:.2f}}<br>Value: %{{y:.2f}}<extra></extra>"
            ),
            row=row_offset + i + 1, col=1
        )
        fig.update_yaxes(title_text=axes[i], row=row_offset + i + 1, col=1)
    fig.update_xaxes(title_text="Time", row=row_offset + 3, col=1)
    return fig

