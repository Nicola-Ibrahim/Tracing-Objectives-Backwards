import plotly.express as px


def get_model_colors(model_names: list[str]) -> dict[str, str]:
    """
    Generates a mapping of model names to unique colors.
    """
    n_models = len(model_names)
    if n_models <= 10:
        colors = px.colors.qualitative.Plotly
    elif n_models <= 24:
        colors = px.colors.qualitative.Dark24
    else:
        colors = px.colors.qualitative.Dark24 + px.colors.qualitative.Plotly

    return {name: colors[idx % len(colors)] for idx, name in enumerate(model_names)}
