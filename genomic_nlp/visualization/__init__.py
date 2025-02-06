"""Visualization utilities for genomic_nlp modules."""

import matplotlib.pyplot as plt  # type: ignore


def set_matplotlib_publication_parameters() -> None:
    """Set matplotlib parameters for publication."""
    plt.rcParams.update(
        {
            "font.size": 7,
            "axes.titlesize": 7,
            "axes.labelsize": 7,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "legend.fontsize": 7,
            "figure.titlesize": 7,
            "figure.dpi": 450,
            "font.sans-serif": ["Arial", "Nimbus Sans"],
            "axes.linewidth": 0.5,
            "xtick.major.width": 0.5,
            "ytick.major.width": 0.5,
            "xtick.minor.width": 0.5,
            "ytick.minor.width": 0.5,
        }
    )


# not implemented yet
# __all__ = [""]
