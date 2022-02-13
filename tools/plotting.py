import math
from typing import List, Optional, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_hist(
    targets: Union[pd.DataFrame, np.ndarray],
    predictions: Optional[Union[pd.DataFrame, np.ndarray]] = None,
    nb_hist_per_line: int = 2,
    columns: Optional[List[str]] = None,
):
    if isinstance(targets, pd.DataFrame):
        if columns is None:
            columns = targets.columns
        targets = targets.to_numpy()
    if isinstance(predictions, pd.DataFrame):
        if columns is None:
            columns = predictions.columns
        predictions = predictions.to_numpy()

    nb_features = targets.shape[1]
    height = int(math.ceil(nb_features / nb_hist_per_line))
    nb_bins = 50
    fig, axs = plt.subplots(
        height,
        nb_hist_per_line,
        figsize=(
            6 * nb_hist_per_line,
            6 * height,
        ),
    )
    for i in range(nb_features):
        axs[i // nb_hist_per_line, i % nb_hist_per_line].hist(
            targets[:, i],
            color="orange",
            bins=nb_bins,
            alpha=0.65,
            weights=np.ones_like(targets[:, i]) / float(len(targets[:, i])),
        )
        # axs[i // nb_hist_per_line, i % nb_hist_per_line].hist(
        #     targets[:, i],
        #     color="red",
        #     bins=nb_bins,
        #     alpha=0.65,
        #     weights=np.ones_like(targets[:, i]) / float(len(targets[:, i])),
        #     cumulative=True,
        #     histtype="step",
        #     linewidth=2
        # )
        if predictions is not None:
            axs[i // nb_hist_per_line, i % nb_hist_per_line].hist(
                predictions[:, i],
                color="blue",
                bins=nb_bins,
                alpha=0.65,
                weights=np.ones_like(predictions[:, i]) / float(len(predictions[:, i])),
            )
        # axs[i // nb_hist_per_line, i % nb_hist_per_line].axvline(
        #     targets[:, i].mean(), color="red", linestyle="dashed", linewidth=3
        # )
        # axs[i // nb_hist_per_line, i % nb_hist_per_line].axvline(
        #     predictions[:, i].mean(), color="purple", linestyle="dashed", linewidth=3
        # )
        if columns is not None:
            axs[i // nb_hist_per_line, i % nb_hist_per_line].set_title(
                f"Histogram of {columns[i]}"
            )
    return fig, axs


def plot_correlation(df):
    fig, ax = plt.subplots(figsize=(15, 10))
    img = ax.matshow(df.corr())
    ax.set_xticks(range(df.select_dtypes(["number"]).shape[1]))
    ax.set_xticklabels(
        df.select_dtypes(["number"]).columns,
        fontsize=14,
        rotation=90,
    )
    ax.set_yticks(range(df.select_dtypes(["number"]).shape[1]))
    ax.set_yticklabels(
        df.select_dtypes(["number"]).columns,
        fontsize=14,
    )
    cb = plt.colorbar(img, ax=ax)
    cb.ax.tick_params(labelsize=14)
    ax.set_title("Correlation Matrix", fontsize=16)
    return fig, ax
