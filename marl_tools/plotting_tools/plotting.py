# python3
# Copyright 2021 InstaDeep Ltd. All rights reserved.
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

from typing import Any, Dict, List, Mapping, Tuple, Optional
import pandas as pd
import colorcet as cc
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.figure import Figure
from rliable import library as rly
from rliable import metrics, plot_utils

"""Tools for plotting MARL experiments based on rliable."""


def performance_profiles(
    dictionary: Mapping[str, Dict[str, Any]],
    metric_name: str,
    metrics_to_normalize: List[str],
) -> Figure:
    """Produces performance profile plots.

    Args:
        dictionary: Dictionary containing 2D arrays of normalised absolute metric scores
            for metric algorithm pairs.
        metric_name: Name of metric to produce plots for.
        metrics_to_normalize: List of metrics that are normalised.

    Returns:
        fig: Matplotlib figure for storing.
    """

    data_dictionary = dictionary[f"mean_{metric_name}"]
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
    dictionary: Mapping[str, Dict[str, Any]],
    metric_name: str,
    metrics_to_normalize: List[str],
    value_round: Optional[int]=2,
    tabular_results_file: Optional[str]="./aggregated_score.csv"
) -> Tuple[Figure, Mapping[str, Mapping[str, int]], Mapping[str, Mapping[str, float]]]:
    """Produces aggregated score plots.

    Args:
        dictionary: Dictionary containing 2D arrays of normalised absolute metric scores
            for metric algorithm pairs.
        metric_name: Name of metric to produce plots for.
        metrics_to_normalize: List of metrics that are normalised.
        value_round:number up to which the results values are rounded
        tabular_results_file: location to store the tabular results

    Returns:
        fig: Matplotlib figure for storing.
        aggregate_scores_dict: Aggregated score values
        aggregate_score_cis_dict: Aggregated score confidence intervals
    """

    data_dictionary = dictionary[f"mean_{metric_name}"]
    algorithms = list(data_dictionary.keys())

    if metric_name in metrics_to_normalize:
        xlabel = "Normalized " + " ".join(metric_name.split("_"))

    else:
        xlabel = " ".join(metric_name.split("_")).capitalize()

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
    
    #Get tabular (csv) results
    tabular_results=aggregate_scores_dict.copy()
    for metric in aggregate_scores_dict.keys():
        for algorithm in aggregate_scores_dict[metric].keys():
            ci=aggregate_score_cis_dict[metric][algorithm]
            value=round(aggregate_scores_dict[metric][algorithm],value_round)

            #get the bootstrap confidence interval
            ci_str="["+str(round(ci[0],value_round))+", "+str(round(ci[1],value_round))+"]"

            result=str(value)+" "+ci_str
            tabular_results[metric][algorithm]=result

    result_csv=pd.DataFrame(aggregate_scores_dict, columns= ['QMIX', 'MADQN',"VDN",'MAPPO'])
    result_csv.to_csv(tabular_results_file, index = False, header=True)
    print("The tabular results are stored in "+tabular_results_file+" and they are the following\n",result_csv)

    return fig, aggregate_scores_dict, aggregate_score_cis_dict


def probability_of_improvement(
    dictionary: Mapping[str, Dict[str, Any]],
    metric_name: str,
    algorithms_to_compare: List[List],
) -> Figure:
    """Produces probability of improvement plots.

    Args:
        dictionary: Dictionary containing 2D arrays of normalised absolute metric scores
            for metric algorithm pairs.
        metric_name: Name of metric to produce plots for.
        algorithms_to_compare: 2D list containing pairs of algorithms to be compared.

    Returns:
        fig: Matplotlib figure for storing.
    """

    data_dictionary = dictionary[f"mean_{metric_name}"]
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
    dictionary: Mapping[str, Dict[str, Any]],
    metric_name: str,
    metrics_to_normalize: List[str],
) -> Tuple[Figure, Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    """Produces sample efficiency curve plots.

    Args:
        dictionary: Dictionary containing 3 dimensional arrays of normalised absolute
             metric scores for metric algorithm pairs.
        metric_name: Name of metric to produce plots for.
        metrics_to_normalize: List of metrics that are normalised.

    Returns:
        fig: Matplotlib figure for storing.
        iqm_scores: IQM score values used in plots.
        iqm_cis: IQM score score confidence intervals used in plots.
    """

    if metric_name in metrics_to_normalize:
        data_dictionary = dictionary[f"mean_norm_{metric_name}"]
    else:
        data_dictionary = dictionary[f"mean_{metric_name}"]

    algorithms = list(data_dictionary.keys())

    if metric_name in metrics_to_normalize:
        ylabel = "Normalized " + " ".join(metric_name.split("_"))

    else:
        ylabel = " ".join(metric_name.split("_")).capitalize()

    frames = np.arange(0, 205, 5)
    frames[-1] = 199

    scores_dict = {
        algorithm: score[:, :, frames] for algorithm, score in data_dictionary.items()
    }

    iqm = lambda scores: np.array(  # noqa: E731
        [metrics.aggregate_iqm(scores[..., frame]) for frame in range(scores.shape[-1])]
    )

    iqm_scores, iqm_cis = rly.get_interval_estimates(scores_dict, iqm, reps=5000)

    fig = plot_utils.plot_sample_efficiency_curve(
        (frames + 1) / 100,
        iqm_scores,
        iqm_cis,
        algorithms=algorithms,
        xlabel=r"Number of timesteps (Millions)",
        ylabel=ylabel,
        legend=algorithms,
        figsize=(15, 8),
        color_palette=cc.glasbey_category10,
    )

    return fig, iqm_scores, iqm_cis
