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

import colorcet as cc
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from rliable import library as rly
from rliable import metrics, plot_utils

"""Tools for plotting MARL experiments based on rliable."""


def performance_profiles(dictionary, metric_name, metrics_to_normalize):

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


def aggregate_scores(dictionary, metric_name, metrics_to_normalize):

    data_dictionary = dictionary[f"mean_{metric_name}"]
    algorithms = list(data_dictionary.keys())

    if metric_name in metrics_to_normalize:
        xlabel = "Normalized " + " ".join(metric_name.split("_"))

    else:
        xlabel = " ".join(metric_name.split("_")).capitalize()

    aggregate_func = lambda x: np.array(
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
    fig, axes = plot_utils.plot_interval_estimates(
        aggregate_scores,
        aggregate_score_cis,
        metric_names=["Median", "IQM", "Mean", "Optimality Gap"],
        algorithms=algorithms,
        xlabel=xlabel,
        color_palette=cc.glasbey_category10,
        xlabel_y_coordinate=-0.5,
    )

    return fig, axes, aggregate_scores, aggregate_score_cis


def probability_of_improvement(dictionary, metric_name, algorithms_to_compare):
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


def sample_efficiency_curves(dictionary, metric_name, metrics_to_normalize):

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

    iqm = lambda scores: np.array(
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
