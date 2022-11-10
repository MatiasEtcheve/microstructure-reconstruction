import math
from typing import Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from sklearn.metrics import mean_absolute_percentage_error

READABLE_NAMES = {}


def plot_overlapping_hist(
    data: List[Union[pd.DataFrame, np.ndarray]],
    nb_hist_per_line: int = 2,
    column_mapping: Dict = {},
    labels: Optional[List[str]] = ["targets", "predictions"],
    ax: Optional[Axes] = None,
) -> Tuple[Figure, Axes]:
    """Plots kernel density estimations of `data`

    Example::
        targets = pd.DataFrame(...,
            columns=["feature_A", "feature_B", "feature_C"],
            index=["observation_1", "observation_2", "observation_3"]
        )
        predictions = pd.DataFrame(...,
            columns=["feature_A", "feature_B", "feature_C"],
            index=["observation_1", "observation_2", "observation_3"]
        )
        fig, axs = plot_kde([targets, predictions], nb_hist_per_line=3)

    Args:
        data (List[pd.DataFrame]): List of DataFrames to plot to plot.
            Each DataFrame in the list must have the same  shape, with the same columns names.
            Each column will be a feature, and a row is an observation
            The plot of each DataFrame will be overlapped.
        nb_hist_per_line (int, optional): Number of histograms to plot per row. Defaults to 2.
        column_mapping (Dict, optional): Mapping between column names in the Dataframe and
            column names to display in the plot. Defaults to {}.
        ax (Axes, optional): Ax which will contain the plot. If None, the ax is created. Defaults to None.
        labels (Optional[List[str]], optional): Labels of each in array in `data`. Defaults to ["targets", "predictions"].

    Returns:
        Axes: ax containing the plots
    """

    features = data[0].columns
    height = int(math.ceil(len(features) / nb_hist_per_line))
    if ax is None:
        fig, ax = plt.subplots(
            height,
            nb_hist_per_line,
            figsize=(
                6 * nb_hist_per_line,
                6 * height,
            ),
        )

    if len(data) < 10:
        colors = ["orange", "blue", "magenta", "green", "cyan", "pink"]
    else:
        colors = np.random.rand(len(data), 3)

    for i, feature in enumerate(list(features)):
        for index, d in enumerate(data):
            sns.histplot(
                data=d.to_numpy()[:, i],
                ax=ax[i // nb_hist_per_line, i % nb_hist_per_line],
                label=labels[index],
                color=colors[index],
            )

            ax[i // nb_hist_per_line, i % nb_hist_per_line].legend()
            ax[i // nb_hist_per_line, i % nb_hist_per_line].set_title(
                f"Histogram of {column_mapping.get(feature, feature)}"
            )
            ax[i // nb_hist_per_line, i % nb_hist_per_line].set_visible(True)
    return ax


def plot_overlapping_kde(
    data: List[Union[pd.DataFrame, np.ndarray]],
    nb_hist_per_line: int = 2,
    column_mapping: Dict = {},
    labels: Optional[List[str]] = ["targets", "predictions"],
    ax: Optional[Axes] = None,
) -> Tuple[Figure, Axes]:
    """Plots kernel density estimations of `data`

    Example::
        targets = pd.DataFrame(...,
            columns=["feature_A", "feature_B", "feature_C"],
            index=["observation_1", "observation_2", "observation_3"]
        )
        predictions = pd.DataFrame(...,
            columns=["feature_A", "feature_B", "feature_C"],
            index=["observation_1", "observation_2", "observation_3"]
        )
        fig, axs = plot_kde([targets, predictions], nb_hist_per_line=3)

    Args:
        data (List[pd.DataFrame]): List of DataFrames to plot to plot.
            Each DataFrame in the list must have the same  shape, with the same columns names.
            Each column will be a feature, and a row is an observation
            The plot of each DataFrame will be overlapped.
        nb_hist_per_line (int, optional): Number of histograms to plot per row. Defaults to 2.
        column_mapping (Dict, optional): Mapping between column names in the Dataframe and
            column names to display in the plot. Defaults to {}.
        ax (Axes, optional): Ax which will contain the plot. If None, the ax is created. Defaults to None.
        labels (Optional[List[str]], optional): Labels of each in array in `data`. Defaults to ["targets", "predictions"].

    Returns:
        Axes: ax containing the plots
    """
    features = data[0].columns
    height = int(math.ceil(len(features) / nb_hist_per_line))
    if ax is None:
        fig, ax = plt.subplots(
            height,
            nb_hist_per_line,
            figsize=(
                6 * nb_hist_per_line,
                6 * height,
            ),
        )
    for i, feature in enumerate(list(features)):
        for index, d in enumerate(data):
            ax[i // nb_hist_per_line, i % nb_hist_per_line] = sns.kdeplot(
                data=d.to_numpy()[:, i],
                shade=True,
                ax=ax[i // nb_hist_per_line, i % nb_hist_per_line],
                label=labels[index],
            )

            ax[i // nb_hist_per_line, i % nb_hist_per_line].legend()
            ax[i // nb_hist_per_line, i % nb_hist_per_line].set_title(
                f"Histogram of {column_mapping.get(feature, feature)}"
            )

    for row in ax:
        for ax in row:
            if not ax.collections:
                ax.set_visible(False)
            index += 1
    return ax


def plot_kde(
    targets: pd.DataFrame,
    predictions: pd.DataFrame,
    column_mapping: Dict[str, str] = {},
    nb_hist_per_line: Optional[int] = 6,
    ax: Optional[Axes] = None,
) -> Axes:
    """Plots the overlapping kernel density estimations of the `targets` and `predictions` DataFrames.

    Args:
        targets (pd.DataFrame): target DataFrame where the columns are the features and the rows are the observations.
        predictions (pd.DataFrame): predictions DataFrame where the columns are the features and the rows are the observations.
        column_mapping (Dict, optional): Mapping between column names in the Dataframe and
            column names to display in the plot. Defaults to {}.
        nb_hist_per_line (int, optional): number of histograms to display per row. Defaults to 6.
        ax (Axes, optional): Ax which will contain the plot. If None, the ax is created. Defaults to None.

    Returns:
        Axes: ax containing the plots
    """
    return plot_overlapping_kde(
        data=[targets, predictions],
        column_mapping=column_mapping,
        labels=["targets", "predictions"],
        nb_hist_per_line=nb_hist_per_line,
        ax=ax,
    )


def plot_hist(
    targets: pd.DataFrame,
    predictions: pd.DataFrame,
    column_mapping: Dict[str, str] = {},
    nb_hist_per_line: Optional[int] = 6,
    ax: Optional[Axes] = None,
) -> Axes:
    """Plots the overlapping kernel density estimations of the `targets` and `predictions` DataFrames.

    Args:
        targets (pd.DataFrame): target DataFrame where the columns are the features and the rows are the observations.
        predictions (pd.DataFrame): predictions DataFrame where the columns are the features and the rows are the observations.
        column_mapping (Dict, optional): Mapping between column names in the Dataframe and
            column names to display in the plot. Defaults to {}.
        nb_hist_per_line (int, optional): number of histograms to display per row. Defaults to 6.
        ax (Axes, optional): Ax which will contain the plot. If None, the ax is created. Defaults to None.

    Returns:
        Axes: ax containing the plots
    """
    return plot_overlapping_hist(
        data=[targets, predictions],
        column_mapping=column_mapping,
        labels=["targets", "predictions"],
        nb_hist_per_line=nb_hist_per_line,
        ax=ax,
    )


def plot_correlation(
    df: pd.DataFrame, column_mapping: Dict = {}, ax: Axes = None
) -> Axes:
    """Plots the correlation between all the features in a DataFrame.

    Args:
        df (pd.DataFrame): DataFrame where the columns are the features and the rows are the observations.
        column_mapping (Dict, optional): Mapping between column names in the Dataframe and
            column names to display in the plot. Defaults to {}.
        ax (Axes, optional): Ax which will contain the plot. If None, the ax is created. Defaults to None.

    Returns:
        Axes: ax containing the plots
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 10))
    img = ax.matshow(df.corr())
    ax.set_xticks(range(df.select_dtypes(["number"]).shape[1]))
    ax.set_xticklabels(
        [
            column_mapping.get(col_name, col_name)
            for col_name in df.select_dtypes(["number"]).columns
        ],
        fontsize=14,
        rotation=90,
    )
    ax.set_yticks(range(df.select_dtypes(["number"]).shape[1]))
    ax.set_yticklabels(
        [
            column_mapping.get(col_name, col_name)
            for col_name in df.select_dtypes(["number"]).columns
        ],
        fontsize=14,
    )
    cb = plt.colorbar(img, ax=ax)
    cb.ax.tick_params(labelsize=14)
    return ax


def plot_horizontal_bars(
    targets: pd.DataFrame,
    predictions: pd.DataFrame,
    column_mapping: Dict[str, str],
    ax: Axes = None,
) -> Axes:
    """Plots the MAE with MAPE annotations.

    Args:
        targets (pd.DataFrame): target DataFrame where the columns are the features and the rows are the observations.
        predictions (pd.DataFrame): predictions DataFrame where the columns are the features and the rows are the observations.
        column_mapping (Dict, optional): Mapping between column names in the Dataframe and
            column names to display in the plot. Defaults to {}.
        ax (Axes, optional): Ax which will contain the plot. If None, the ax is created. Defaults to None.

    Returns:
        Axes: ax containing the plots
    """

    def show_values_on_bars(axs, values):
        def _show_on_single_plot(ax):
            i = 0
            index_patch = 0
            for index, p in enumerate(ax.lines):
                patch = ax.patches[index_patch]
                if index % 3 == 2:
                    _x = p.get_xdata()[-1] + ax.get_xlim()[1] / 50
                    _y = patch.get_y() + patch.get_height() / 2
                    value = "MAPE = {0:.3g}%".format(values[i])
                    i += 1
                    index_patch += 1
                    ax.text(
                        _x,
                        _y,
                        value,
                        ha="left",
                        va="center",
                        fontsize=min(220 / len(ax.patches), 10),
                    )

        if isinstance(axs, np.ndarray):
            for idx, ax in np.ndenumerate(axs):
                _show_on_single_plot(ax)
        else:
            _show_on_single_plot(axs)

    predictions_with_correct_names = predictions.copy().rename(column_mapping, axis=1)
    targets_with_correct_names = targets.copy().rename(column_mapping, axis=1)

    mae = pd.DataFrame(
        np.abs(predictions_with_correct_names - targets_with_correct_names),
        columns=predictions_with_correct_names.columns,
    )
    mape = pd.DataFrame(
        100
        * np.expand_dims(
            mean_absolute_percentage_error(
                targets_with_correct_names,
                predictions_with_correct_names,
                multioutput="raw_values",
            ),
            0,
        ),
        columns=predictions_with_correct_names.columns,
    )
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(10, 10))

    sns.barplot(
        data=mae,
        ci=95,
        capsize=0.2,
        orient="h",
        ax=ax,
    )
    show_values_on_bars(
        ax,
        np.squeeze(mape.to_numpy()),
    )
    return ax
