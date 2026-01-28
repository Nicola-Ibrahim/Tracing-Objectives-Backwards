"""
Configuration and layout constants for the 2D visualizer.
"""

DIAGNOSTIC_EXPLANATIONS = [
    (
        0.75,
        "<b>Model Surfaces</b>: Predicted decision surface (ghost scatter for probabilistic models) vs data points. <i>Goal</i>: Scatter cloud should cover data distribution.",
    ),
    (
        0.20,
        "<b>Learning Curves</b>: Loss over epochs. <i>Goal</i>: Decrease and converge. Gap = Overfitting.",
    ),
]


def format_coordinate_symbol(symbol: str, idx: int) -> str:
    """Format symbol with subscript for cleaner visualization."""
    subscripts = {1: "\u2081", 2: "\u2082"}
    return f"{symbol}{subscripts.get(idx, idx)}"


def get_subplot_titles(input_symbol: str, output_symbol: str) -> list[str]:
    """Generate standardized subplot titles for the 2D visualizer."""
    s1 = format_coordinate_symbol(input_symbol, 1)
    s2 = format_coordinate_symbol(input_symbol, 2)
    o1 = format_coordinate_symbol(output_symbol, 1)
    o2 = format_coordinate_symbol(output_symbol, 2)

    return [
        f"{o1}({s1}, {s2})",
        f"{o2}({s1}, {s2})",
        "Training / Validation / Test",
    ]
