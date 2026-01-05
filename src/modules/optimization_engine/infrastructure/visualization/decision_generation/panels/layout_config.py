import plotly.express as px

# Layout Constants
MAX_COLS = 3
DEFAULT_WIDTH = 1800
DEFAULT_PLOT_HEIGHT = 600

# Styling Constants
PARETO_MARKER = dict(color="lightgray", size=5, opacity=0.3)
TARGET_MARKER = dict(
    color="red",
    symbol="star",
    size=15,
    line=dict(width=2, color="black"),
)


def select_palette(n_models: int) -> list[str]:
    """Select an appropriate qualitative color palette based on model count."""
    if n_models <= 10:
        return px.colors.qualitative.Plotly
    if n_models <= 24:
        return px.colors.qualitative.Dark24
    return px.colors.qualitative.Dark24 + px.colors.qualitative.Plotly
