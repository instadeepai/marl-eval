# python3
# Copyright 2022 InstaDeep Ltd. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Any, Dict, Optional

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from rliable.plot_utils import _annotate_and_decorate_axis


def plot_single_task_curve(
    aggregated_data: Dict[str, Any],
    algorithms: list,
    colors: Optional[Dict] = None,
    color_palette: str = "colorblind",
    figsize: tuple = (7, 5),
    xlabel: str = "Number of Frames (in millions)",
    ylabel: str = "Aggregate Human Normalized Score",
    ax: Optional[Axes] = None,
    labelsize: str = "xx-large",
    ticklabelsize: str = "xx-large",
    legend_map: Optional[Dict] = None,
    run_times: Optional[Dict] = None,
    **kwargs: Any,
) -> Figure:
    """Plots an aggregate metric with CIs as a function of environment frames.

    Args:
      aggregated_data: Dictionary containing the mean and 95% CI at each
        evaluation step for all algorithms on a particular task.
      algorithms: List of methods used for plotting. If None, defaults to all the
        keys in `point_estimates`.
      colors: Dictionary that maps each algorithm to a color. If None, then this
        mapping is created based on `color_palette`.
      color_palette: `seaborn.color_palette` object for mapping each method to a
        color.
      figsize: Size of the figure passed to `matplotlib.subplots`. Only used when
        `ax` is None.
      xlabel: Label for the x-axis.
      ylabel: Label for the y-axis.
      ax: `matplotlib.axes` object.
      labelsize: Font size of the x-axis label.
      ticklabelsize: Font size of the ticks.
      legend_map: Dictionary that maps each algorithm to a label in the legend.
        If None, then this mapping is created based on `algorithms`.
      run_times: Dictionary that maps each algorithm to the number of seconds it
        took to run. If None, then environment steps will be displayed.
      **kwargs: Arbitrary keyword arguments.

    Returns:
      `axes.Axes` object containing the plot.
    """
    extra_info = aggregated_data.pop("extra")

    if ax is None:
        _, ax = plt.subplots(figsize=figsize)
    if algorithms is None:
        algorithms = list(aggregated_data.keys())
    if colors is None:
        color_palette = sns.color_palette(color_palette, n_colors=len(algorithms))
        colors = dict(zip(algorithms, color_palette))

    marker = kwargs.pop("marker", "o")
    linewidth = kwargs.pop("linewidth", 2)

    for algorithm in algorithms:
        x_axis_len = len(aggregated_data[algorithm]["mean"])

        # Set x-axis values to match evaluation interval steps.
        x_axis_values = np.arange(x_axis_len) * extra_info["evaluation_interval"]

        if run_times is not None:
            x_axis_values = np.linspace(0, run_times[algorithm] / 60, x_axis_len)

        metric_values = np.array(aggregated_data[algorithm]["mean"])
        confidence_interval = np.array(aggregated_data[algorithm]["ci"])
        lower, upper = (
            metric_values - confidence_interval,
            metric_values + confidence_interval,
        )

        if legend_map is not None:
            algorithm_name = legend_map[algorithm]
        else:
            algorithm_name = algorithm

        ax.plot(
            x_axis_values,
            metric_values,
            color=colors[algorithm],
            marker=marker,
            linewidth=linewidth,
            label=algorithm_name,
        )
        ax.fill_between(
            x_axis_values, y1=lower, y2=upper, color=colors[algorithm], alpha=0.2
        )

    return _annotate_and_decorate_axis(
        ax,
        xlabel=xlabel,
        ylabel=ylabel,
        labelsize=labelsize,
        ticklabelsize=ticklabelsize,
        **kwargs,
    )
