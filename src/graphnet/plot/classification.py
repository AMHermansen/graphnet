"""Default plots for classification tasks."""
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Union

from sklearn.metrics import roc_curve, auc

from graphnet.plot.utils import (
    LogitHistData,
    ROCCurveData,
)


def default_hist_plot(
    cache: pd.DataFrame,
    prediction_labels: List[str],
    target_ids: pd.Series,
    output_dir: Union[str, Path],
    style: Union[List[str], dict, Path, List],
    logit_hist_data: LogitHistData,
    file_suffix: str = "",
    file_prefix: str = "",
    y_log: bool = False,
) -> None:
    """Create histogram of logits for each prediction label.

    Args:
        cache: Cache DataFrame containing predictions.
        prediction_labels: Labels indicating which columns are predictions.
        target_ids: Primary target indexes.
        output_dir: Output directory.
        style: Path to StyleSheet for plot.
        logit_hist_data: LogitHistData dataclass object.
        file_suffix: String appended to filename.
        file_prefix: String prepended to filename.
        y_log: If True plot y-axis on log scale.

    Returns: None
    """
    with (plt.style.context(style)):
        fig, ax = plt.subplots()
        for index, pred_label in enumerate(prediction_labels):
            logits = np.log(
                (
                    p_eps := (
                        cache[target_ids == index][pred_label]
                        - 2 * logit_hist_data.epsilon
                    )
                )
                / (1 - p_eps)
            ).dropna()
            ax.hist(
                logits,
                bins=logit_hist_data.bins,
                histtype=logit_hist_data.hist_type,
                density=logit_hist_data.density,
                alpha=logit_hist_data.alpha,
                label=f"{pred_label}_logits",
            )
        ax.set_xlabel(logit_hist_data.xlabel)  # ignore
        ax.set_ylabel(logit_hist_data.ylabel)
        if y_log:
            ax.set_yscale("log")
        ax.legend()
        maybe_log = "log_" if y_log else ""
        fig.savefig(
            Path(output_dir)
            / f"{file_prefix}{maybe_log}logit_hist{file_suffix}.png",
        )
        plt.close(fig)


def default_roc_curve(
    predictions: pd.DataFrame,
    target_ids: pd.Series,
    output_dir: Union[str, Path],
    style: Union[List[str], dict, Path, List],
    roc_curve_data: ROCCurveData,
    file_suffix: str = "",
    file_prefix: str = "",
    y_log: bool = False,
) -> None:
    """Make ROC curve plot.

    Args:
        predictions: Predictions from model.
        target_ids: Target indexes.
        output_dir: Output directory.
        style: Path to a preferred stylesheet for the plot.
        roc_curve_data: ROC curve dataclass object.
        file_suffix: String appended to the output filename.
        file_prefix: String prepended to the output filename.
        y_log: If True, plot y-axis on log scale.

    Returns: None
    """
    with (plt.style.context(style)):
        fig, ax = plt.subplots()
        for index, (column, legend_label) in enumerate(
            zip(predictions.columns, roc_curve_data.legend_labels)
        ):
            prediction = predictions[column]
            fpr, tpr, _ = roc_curve(
                target_ids,
                prediction,
                pos_label=index,
            )
            roc_auc = auc(fpr, tpr)
            ax.plot(
                fpr,
                tpr,
                label=f"{legend_label}\nauc={roc_auc:{roc_curve_data.auc_format}}",
            )
        ax.plot(
            np.linspace(1e-08, 1, 100),
            np.linspace(1e-08, 1, 100),
            color="k",
            linestyle="--",
            linewidth=0.75,
            label="fpr=tpr",
        )
        ax.set_xlabel(roc_curve_data.xlabel)
        ax.set_ylabel(roc_curve_data.ylabel)
        if y_log:
            ax.set_yscale("log")
        ax.legend()
        maybe_log = "log_" if y_log else ""
        fig.savefig(
            Path(output_dir) / f"{file_prefix}{maybe_log}roc{file_suffix}.png",
        )
        plt.close(fig)
