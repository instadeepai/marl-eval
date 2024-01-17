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

from typing import Any, Dict, List, Optional, Tuple

import colorcet as cc
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.figure import Figure
from rliable import library as rly
from rliable import metrics, plot_utils

from marl_eval.plotting_tools.plot_utils import plot_single_task_curve
from marl_eval.utils.data_processing_utils import (
    get_and_aggregate_data_single_task,
    lower_case_inputs,
)

"""Tools for plotting MARL experiments based on rliable."""


def performance_profiles(
    dictionary: Dict[str, Dict[str, Any]],
    metric_name: str,
    metrics_to_normalize: List[str],
    legend_map: Optional[Dict[str, str]] = None,
) -> Figure:
    """Produces performance profile plots.

    Args:
        dictionary: Dictionary containing 2D arrays of normalised absolute metric scores
            for metric algorithm pairs.
        metric_name: Name of metric to produce plots for.
        metrics_to_normalize: List of metrics that are normalised.
        legend_map: Dictionary that maps each algorithm to a custom legend label.

    Returns:
        fig: Matplotlib figure for storing.
    """

    metric_name, metrics_to_normalize = lower_case_inputs(
        metric_name, metrics_to_normalize
    )

    if metric_name in metrics_to_normalize:
        data_dictionary = dictionary[f"mean_norm_{metric_name}"]
    else:
        data_dictionary = dictionary[f"mean_{metric_name}"]

    # Upper case all algorithm names
    upper_algo_dict = {algo.upper(): value for algo, value in data_dictionary.items()}
    data_dictionary = upper_algo_dict
    algorithms = list(data_dictionary.keys())

    if legend_map is not None:
        legend_map = {algo.upper(): value for algo, value in legend_map.items()}
        # Replace keys in data dict with corresponding key in legend map
        data_dictionary = {
            legend_map[algo]: value for algo, value in data_dictionary.items()
        }
        algorithms = list(data_dictionary.keys())

    if metric_name in metrics_to_normalize:
        xlabel = "Normalized " + " ".join(metric_name.split("_"))

    else:
        xlabel = " ".join(metric_name.split("_")).capitalize()

    score_distributions, score_distributions_cis = rly.create_performance_profile(
        data_dictionary, np.linspace(0, 1, 100)
    )

    # Plot score distributions
    fig, ax = plt.subplots(ncols=1, figsize=(7, 5))
    plot_utils.plot_performance_profiles(
        score_distributions,
        np.linspace(0, 1, 100),
        performance_profile_cis=score_distributions_cis,
        colors=dict(zip(algorithms, sns.color_palette(cc.glasbey_category10))),
        xlabel=f"{xlabel} " + r"$(\tau)$",
        ax=ax,
        legend=algorithms,
    )
    return fig


def aggregate_scores(
    dictionary: Dict[str, Dict[str, Any]],
    metric_name: str,
    metrics_to_normalize: List[str],
    rounding_decimals: Optional[int] = 2,
    tabular_results_file_path: str = "./aggregated_score",
    save_tabular_as_latex: Optional[bool] = False,
    legend_map: Optional[Dict[str, str]] = None,
) -> Tuple[Figure, Dict[str, Dict[str, int]], Dict[str, Dict[str, float]]]:
    """Produces aggregated score plots.

    Args:
        dictionary: Dictionary containing 2D arrays of normalised absolute metric scores
            for metric algorithm pairs.
        metric_name: Name of metric to produce plots for.
        metrics_to_normalize: List of metrics that are normalised.
        rounding_decimals:number up to which the results values are rounded.
        tabular_results_file_path: location to store the tabular results.
        save_tabular_as_latex: store tabular results in latex format in a .txt file.
        legend_map: Dictionary that maps each algorithm to a custom legend label.

    Returns:
        fig: Matplotlib figure for storing.
        aggregate_scores_dict: Aggregated score values
        aggregate_score_cis_dict: Aggregated score confidence intervals
    """

    metric_name, metrics_to_normalize = lower_case_inputs(
        metric_name, metrics_to_normalize
    )

    if metric_name in metrics_to_normalize:
        data_dictionary = dictionary[f"mean_norm_{metric_name}"]
        xlabel = "Normalized " + " ".join(metric_name.split("_"))
    else:
        data_dictionary = dictionary[f"mean_{metric_name}"]
        xlabel = " ".join(metric_name.split("_")).capitalize()

    # Upper case all algorithm names
    upper_algo_dict = {algo.upper(): value for algo, value in data_dictionary.items()}
    data_dictionary = upper_algo_dict
    algorithms = list(data_dictionary.keys())

    if legend_map is not None:
        legend_map = {algo.upper(): value for algo, value in legend_map.items()}
        # Replace keys in data dict with corresponding key in legend map
        data_dictionary = {
            legend_map[algo]: value for algo, value in data_dictionary.items()
        }
        algorithms = list(data_dictionary.keys())

    aggregate_func = lambda x: np.array(  # noqa: E731
        [
            metrics.aggregate_median(x),
            metrics.aggregate_iqm(x),
            metrics.aggregate_mean(x),
            metrics.aggregate_optimality_gap(x),
        ]
    )
    aggregate_scores, aggregate_score_cis = rly.get_interval_estimates(
        data_dictionary, aggregate_func, reps=50000
    )

    metric_names = ["Median", "IQM", "Mean", "Optimality Gap"]

    fig, axes = plot_utils.plot_interval_estimates(
        aggregate_scores,
        aggregate_score_cis,
        metric_names=metric_names,
        algorithms=algorithms,
        xlabel=xlabel,
        color_palette=cc.glasbey_category10,
        xlabel_y_coordinate=-0.5,
    )

    # Reformat aggregate scores and aggregate score
    # confidences interval as dictionaries for easier use.
    aggregate_scores_dict = dict()
    for algorithm, scores in aggregate_scores.items():
        algorithm_scores_dict = dict()
        for metric, metric_value in zip(metric_names, scores):
            algorithm_scores_dict[metric] = metric_value
        aggregate_scores_dict[algorithm] = algorithm_scores_dict

    aggregate_score_cis_dict = dict()
    for algorithm, scores in aggregate_score_cis.items():
        algorithm_cis_dict = dict()
        for metric_value_idx, metric in enumerate(metric_names):
            algorithm_cis_dict[metric] = scores[:, metric_value_idx]
        aggregate_score_cis_dict[algorithm] = algorithm_cis_dict

    # Get tabular (csv) results
    tabular_results = aggregate_scores_dict.copy()
    algorithms = list(aggregate_scores_dict.keys())

    for algorithm in aggregate_scores_dict.keys():
        for metric in aggregate_scores_dict[algorithm].keys():
            ci = aggregate_score_cis_dict[algorithm][metric]
            value = round(aggregate_scores_dict[algorithm][metric], rounding_decimals)

            # get the bootstrap confidence interval
            ci_str = (
                "["
                + str(round(ci[0], rounding_decimals))
                + ", "
                + str(round(ci[1], rounding_decimals))
                + "]"
            )

            result = str(value) + " " + ci_str
            tabular_results[algorithm][metric] = result

    tabular_results_df = pd.DataFrame(tabular_results, columns=algorithms)
    tabular_results_df.to_csv(
        tabular_results_file_path + "_" + metric_name + ".csv", index=False, header=True
    )
    print(
        "The tabular results are stored in "
        + tabular_results_file_path
        + "_"
        + metric_name
        + ".csv"
        + " and they are the following\n",
        tabular_results_df,
    )

    if save_tabular_as_latex:
        with open(
            tabular_results_file_path + "_" + metric_name + "_latex.txt", "a"
        ) as f:
            print(tabular_results_df.style.to_latex(), file=f)
            print(
                "The latex tabular results are stored in "
                + tabular_results_file_path
                + "_"
                + metric_name
                + "_latex.txt"
            )

    return fig, aggregate_scores_dict, aggregate_score_cis_dict


def probability_of_improvement(
    dictionary: Dict[str, Dict[str, Any]],
    metric_name: str,
    metrics_to_normalize: List[str],
    algorithms_to_compare: List[List],
    legend_map: Optional[Dict[str, str]] = None,
) -> Figure:
    """Produces probability of improvement plots.

    Args:
        dictionary: Dictionary containing 2D arrays of normalised absolute metric scores
            for metric algorithm pairs.
        metric_name: Name of metric to produce plots for.
        metrics_to_normalize: List of metrics that are normalised.
        algorithms_to_compare: 2D list containing pairs of algorithms to be compared.
        legend_map: Dictionary that maps each algorithm to a custom legend label.

    Returns:
        fig: Matplotlib figure for storing.
    """

    metric_name, metrics_to_normalize = lower_case_inputs(
        metric_name, metrics_to_normalize
    )

    if metric_name in metrics_to_normalize:
        data_dictionary = dictionary[f"mean_norm_{metric_name}"]
    else:
        data_dictionary = dictionary[f"mean_{metric_name}"]

    # Upper case all algorithm names
    upper_algo_dict = {algo.upper(): value for algo, value in data_dictionary.items()}
    data_dictionary = upper_algo_dict

    # Upper case algorithm pairs
    algorithms_to_compare = [
        [pair[0].upper(), pair[1].upper()] for pair in algorithms_to_compare
    ]

    if legend_map is not None:
        legend_map = {algo.upper(): value for algo, value in legend_map.items()}
        # Replace keys in data dict with corresponding key in legend map
        data_dictionary = {
            legend_map[algo]: value for algo, value in data_dictionary.items()
        }
        # Make sure that the algorithms to compare are also in the legend map
        algorithms_to_compare = [
            [legend_map[pair[0]], legend_map[pair[1]]] for pair in algorithms_to_compare
        ]

    algorithm_pairs = {}
    for pair in algorithms_to_compare:
        algorithm_pairs[",".join(pair)] = (
            data_dictionary[pair[0]],
            data_dictionary[pair[1]],
        )
    average_probabilities, average_prob_cis = rly.get_interval_estimates(
        algorithm_pairs, metrics.probability_of_improvement, reps=2000
    )
    fig = plot_utils.plot_probability_of_improvement(
        average_probabilities, average_prob_cis, color_palette=cc.glasbey_category10
    )
    return fig


def sample_efficiency_curves(
    dictionary: Dict[str, Dict[str, Any]],
    metric_name: str,
    metrics_to_normalize: List[str],
    legend_map: Optional[Dict[str, str]] = None,
    xlabel: str = "Timesteps",
) -> Tuple[Figure, Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    """Produces sample efficiency curve plots.

    Args:
        dictionary: Dictionary containing 3 dimensional arrays of normalised absolute
             metric scores for metric algorithm pairs.
        metric_name: Name of metric to produce plots for.
        metrics_to_normalize: List of metrics that are normalised.
        legend_map: Dictionary that maps each algorithm to a custom legend label.
        xlabel: Label for x-axis.

    Returns:
        fig: Matplotlib figure for storing.
        iqm_scores: IQM score values used in plots.
        iqm_cis: IQM score score confidence intervals used in plots.
    """

    metric_name, metrics_to_normalize = lower_case_inputs(
        metric_name, metrics_to_normalize
    )

    # Extract the extra info
    extra = dictionary.pop("extra")

    if metric_name in metrics_to_normalize:
        data_dictionary = dictionary[f"mean_norm_{metric_name}"]
        ylabel = "Normalized " + " ".join(metric_name.split("_"))
    else:
        data_dictionary = dictionary[f"mean_{metric_name}"]
        ylabel = " ".join(metric_name.split("_")).capitalize()

    # Upper case all algorithm names
    upper_algo_dict = {algo.upper(): value for algo, value in data_dictionary.items()}
    data_dictionary = upper_algo_dict
    algorithms = list(data_dictionary.keys())

    if legend_map is not None:
        legend_map = {algo.upper(): value for algo, value in legend_map.items()}
        # Replace keys in data dict with corresponding key in legend map
        data_dictionary = {
            legend_map[algo]: value for algo, value in data_dictionary.items()
        }
        algorithms = list(data_dictionary.keys())

    # Find lowest values from amount of runs that have completed
    # across all algorithms
    run_lengths = [data_dictionary[algo].shape[2] for algo in data_dictionary]
    min_run_length = np.min(run_lengths)

    frames = np.arange(0, min_run_length, 1)

    # Create x-axis values that match evaluation step intervals.
    x_axis_values = frames * extra["evaluation_interval"]

    scores_dict = {
        algorithm: score[:, :, frames] for algorithm, score in data_dictionary.items()
    }

    iqm = lambda scores: np.array(  # noqa: E731
        [metrics.aggregate_iqm(scores[..., frame]) for frame in range(scores.shape[-1])]
    )

    iqm_scores, iqm_cis = rly.get_interval_estimates(scores_dict, iqm, reps=5000)

    fig = plot_utils.plot_sample_efficiency_curve(
        x_axis_values,
        iqm_scores,
        iqm_cis,
        algorithms=algorithms,
        xlabel=xlabel,
        ylabel=ylabel,
        legend=algorithms,
        figsize=(15, 8),
        color_palette=cc.glasbey_category10,
    )

    dictionary["extra"] = extra

    return fig, iqm_scores, iqm_cis


def plot_single_task(
    processed_data: Dict[str, Dict[str, Any]],
    environment_name: str,
    task_name: str,
    metric_name: str,
    metrics_to_normalize: List[str],
    xlabel: str = "Timesteps",
    legend_map: Optional[Dict[str, str]] = None,
    run_times: Optional[Dict[str, float]] = None,
) -> Figure:
    """Produces aggregated plot for a single task in an environment.

    Args:
        processed_data: Dictionary containing processed data.
        environment_name: Name of environment to produce plots for.
        task_name: Name of task to produce plots for.
        metric_name: Name of metric to produce plots for.
        metrics_to_normalize: List of metrics that are normalised.
        xlabel: Label for x-axis.
        legend_map: Dictionary that maps each algorithm to a label in the legend.
            If None, then this mapping is created based on `algorithms`.
        run_times: Dictionary that maps each algorithm to the number of seconds it
            took to run. If None, then environment steps will be displayed.
    """
    metric_name, task_name, environment_name, metrics_to_normalize = lower_case_inputs(
        metric_name, task_name, environment_name, metrics_to_normalize
    )

    task_mean_ci_data = get_and_aggregate_data_single_task(
        processed_data=processed_data,
        environment_name=environment_name,
        metric_name=metric_name,
        task_name=task_name,
        metrics_to_normalize=metrics_to_normalize,
    )

    if metric_name in metrics_to_normalize:
        ylabel = "Normalized " + " ".join(metric_name.split("_"))
    else:
        ylabel = " ".join(metric_name.split("_")).capitalize()

    # Upper case all algorithm names
    upper_algo_dict = {
        (algo.upper() if algo != "extra" else algo): value
        for algo, value in task_mean_ci_data.items()
    }
    task_mean_ci_data = upper_algo_dict
    algorithms = list(task_mean_ci_data.keys())
    algorithms.remove("extra")

    if legend_map is not None:
        legend_map = {algo.upper(): value for algo, value in legend_map.items()}

    if run_times is not None:
        run_times = {algo.upper(): value for algo, value in run_times.items()}
        xlabel = "Time (Minutes)"

    fig = plot_single_task_curve(
        task_mean_ci_data,
        algorithms=algorithms,
        xlabel=xlabel,
        ylabel=ylabel,
        legend=algorithms,
        figsize=(15, 8),
        color_palette=cc.glasbey_category10,
        legend_map=legend_map,
        run_times=run_times,
        marker="",
    )

    return fig
