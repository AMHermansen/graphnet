"""Module containing Utility (data)classes for plotting."""
from pathlib import Path
from typing import List
import numpy as np
from attrs import define
from matplotlib import pyplot as plt


class LogitHistData:
    """Data for plotting logit histograms."""

    def __init__(
        self,
        epsilon: float = 1e-6,
        bins: np.ndarray = np.linspace(-20, 20, 81),
        hist_type: str = "step",
        density: bool = True,
        alpha: float = 0.5,
        xlabel: str = "logit",
        ylabel: str = "density",
        titel: str = "Logit Histogram",
    ):
        """Construct ´LogitHistData´ object.

        Args:
            epsilon: Epsilon value for numerical stability of logits.
            bins: Bins in histogram.
            hist_type: Histogram type.
            density: If true plot density instead of counts.
            alpha: Transparency of histogram.
            xlabel: Label on x-axis.
            ylabel: Label on y-axis.
            titel: Title of plot.
        """
        self.epsilon = epsilon
        self.bins = bins
        self.hist_type = hist_type
        self.density = density
        self.alpha = alpha
        self.xlabel = xlabel
        self.ylabel = ylabel
        self.titel = titel


class ROCCurveData:
    """Data for plotting ROC curves."""

    def __init__(
        self,
        legend_labels: List[str],
        alpha: float = 0.5,
        xlabel: str = "False Positive Rate",
        ylabel: str = "True Positive Rate",
        titel: str = "ROC Curve",
        auc_format: str = ".5f",
    ):
        """Construct ´ROCCurveData´ containing data for plotting ROC curves.

        Args:
            legend_labels: Legend labels in ROC curve.
            alpha: Transparency of ROC curve.
            xlabel: X axis label.
            ylabel: Y axis label.
            titel: Title of plot.
            auc_format: Float-format of AUC value.
        """
        self.legend_labels = legend_labels
        self.alpha = alpha
        self.xlabel = xlabel
        self.ylabel = ylabel
        self.titel = titel
        self.auc_format = auc_format
