import plotly.colors
import plotly.express as px


def _to_glassy_rgba(color: str, opacity: float = 0.6) -> str:
    """
    Converts a color (hex or rgb) to RGBA format with transparency.
    """
    # If already rgba, just return it
    if color.startswith("rgba"):
        return color

    try:
        # Convert any plotly color to RGB tuple
        rgb = plotly.colors.unlabel_rgb(
            plotly.colors.convert_colors_to_same_type([color], "rgb")[0][0]
        )
        return f"rgba({rgb[0]}, {rgb[1]}, {rgb[2]}, {opacity})"
    except Exception:
        return color


def darken_rgba(rgba_color: str, factor: float = 0.7) -> str:
    """
    Darkens an RGBA color by reducing RGB components.
    Expects format: rgba(r, g, b, a)
    """
    if not rgba_color.startswith("rgba"):
        return rgba_color

    try:
        # Extract values
        content = rgba_color.replace("rgba(", "").replace(")", "")
        parts = [float(x.strip()) for x in content.split(",")]
        if len(parts) < 4:
            return rgba_color

        r, g, b, a = parts
        # Darken RGB, keep alpha at 1.0 for solid border or at original alpha?
        # User said "same color but darker", typically borders are more opaque.
        return f"rgba({int(r * factor)}, {int(g * factor)}, {int(b * factor)}, 1.0)"
    except Exception:
        return rgba_color


def get_model_colors(model_names: list[str]) -> dict[str, str]:
    """
    Generates a mapping of model names to unique glassy colors.
    """
    n_models = len(model_names)
    if n_models <= 10:
        colors = px.colors.qualitative.Plotly
    elif n_models <= 24:
        colors = px.colors.qualitative.Dark24
    else:
        colors = px.colors.qualitative.Dark24 + px.colors.qualitative.Plotly

    return {
        name: _to_glassy_rgba(colors[idx % len(colors)])
        for idx, name in enumerate(model_names)
    }
