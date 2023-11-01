"""Module containing Utility (data)classes for plotting."""
from pathlib import Path
from typing import List
import numpy as np
from attrs import define
from matplotlib import pyplot as plt


@define
class LogitHistData:
    """Data for plotting logit histograms."""

    epsilon: float = 1e-6
    bins: np.ndarray = np.linspace(-20, 20, 81)
    hist_type: str = "step"
    density: bool = True
    alpha: float = 0.5
    xlabel: str = "logit"
    ylabel: str = "density"
    titel: str = "Logit Histogram"


@define
class ROCCurveData:
    """Data for plotting ROC curves."""

    legend_labels: List[str]
    alpha: float = 0.5
    xlabel: str = "False Positive Rate"
    ylabel: str = "True Positive Rate"
    titel: str = "ROC Curve"
    auc_format: str = ".5f"
