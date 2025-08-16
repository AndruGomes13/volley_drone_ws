from plotly import graph_objects as go
from analysis_utils import ACFAnalysis



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