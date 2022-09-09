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

import copy
from typing import Any, Dict, List, Tuple

import numpy as np

"""Tools for processing MARL experiment data."""


def data_process_pipeline(
    raw_data: Dict[str, Any], metrics_to_normalize: List[str]
) -> Dict[str, Any]:
    """Function for processing raw input experiment data.

    Args:
        raw_data: Dictionary containing raw data that was read in
            from JSON file.
        metrics_to_normalize: A list of metric names for metrics that should
            be min/max normalised. These metric names should match the names as
            given in the raw dataset

    Returns:
        processed_data: Dictionary containing processed experiment data where relevant
            metrics have been min/max normalised and the mean of all arrays in the dataset
            have been computed and added to the dataset.
    """

    metric_min_max_info = {}

    # Create global max, min dictionary
    for metric in metrics_to_normalize:
        for env in raw_data.keys():
            metric_min_max_info[env] = {}
            for scenario in raw_data[env]:
                metric_min_max_info[env][scenario] = {}
                for algorithm in raw_data[env][scenario]:
                    metric_min_max_info[env][scenario][algorithm] = {}
                    metric_min_max_info[env][scenario][algorithm][metric] = {}
                    metric_min_max_info[env][scenario][algorithm][metric][
                        "global_min"
                    ] = 1_000_000
                    metric_min_max_info[env][scenario][algorithm][metric][
                        "global_max"
                    ] = -1_000_000

    # Now we have to traverse the data to find the global min and max
    for metric in metrics_to_normalize:
        for env in raw_data.keys():
            for scenario in raw_data[env]:
                for algorithm in raw_data[env][scenario]:
                    for run in raw_data[env][scenario][algorithm]:
                        # exclude final step since it contains the absolute metrics
                        for step in list(
                            raw_data[env][scenario][algorithm][run].keys()
                        )[:-1]:
                            min_val = np.min(
                                raw_data[env][scenario][algorithm][run][step][metric]
                            )
                            max_val = np.max(
                                raw_data[env][scenario][algorithm][run][step][metric]
                            )
                            if (
                                min_val
                                < metric_min_max_info[env][scenario][algorithm][metric][
                                    "global_min"
                                ]
                            ):
                                metric_min_max_info[env][scenario][algorithm][metric][
                                    "global_min"
                                ] = min_val
                            if (
                                max_val
                                > metric_min_max_info[env][scenario][algorithm][metric][
                                    "global_max"
                                ]
                            ):
                                metric_min_max_info[env][scenario][algorithm][metric][
                                    "global_max"
                                ] = max_val

    # there are definitely more elegant and faster ways of doing exactly this with
    # the tree library. but this is just a POC

    processed_data = copy.deepcopy(raw_data)

    for env in raw_data.keys():
        for scenario in raw_data[env]:
            for algorithm in raw_data[env][scenario]:
                for run in raw_data[env][scenario][algorithm]:
                    for step in raw_data[env][scenario][algorithm][run]:
                        for metric in raw_data[env][scenario][algorithm][run][step]:
                            if metric.split("_")[0].lower() != "step":
                                mean = np.mean(
                                    raw_data[env][scenario][algorithm][run][step][
                                        metric
                                    ]
                                )
                                processed_data[env][scenario][algorithm][run][step][
                                    f"mean_{metric}"
                                ] = mean
                                if metric in metrics_to_normalize:
                                    metric_array = np.array(
                                        raw_data[env][scenario][algorithm][run][step][
                                            metric
                                        ]
                                    )
                                    metric_global_min = metric_min_max_info[env][
                                        scenario
                                    ][algorithm][metric]["global_min"]
                                    metric_global_max = metric_min_max_info[env][
                                        scenario
                                    ][algorithm][metric]["global_max"]
                                    normed_metric_array = (
                                        metric_array - metric_global_min
                                    ) / (metric_global_max - metric_global_min)
                                    processed_data[env][scenario][algorithm][run][step][
                                        f"norm_{metric}"
                                    ] = normed_metric_array.tolist()
                                    processed_data[env][scenario][algorithm][run][step][
                                        f"mean_norm_{metric}"
                                    ] = np.mean(normed_metric_array)

    return processed_data


def create_matrices_for_rliable(
    data_dictionary: Dict[str, Any],
    environment_name: str,
    metrics_to_normalize: List[str],
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Create dictionary containing matrices required for using the rliable tools.
        These are a (number of runs x number of tasks) array where the entries are
        the normalised absolute metrics for a given environments. And a dictionary
        with a (number of logging steps x number of runs x number of tasks) array
        as values and environment names as keys to be used for computing sample
        efficiency curves.


    Args:
        data_dictionary: Dictionary of data that has been processed
        environment_name: Name of environment for which matrices should be
            computed.
        metrics_to_normalize: LIst of metric names of metrics that should be
            normalised.

    Returns:
        metric_dictionary_return: dictionary containing normalised matrices
            to be used by rliable tools. The dictionary keys are the metrics for
            for which normalised arrays were computed at the algorithm level.
        final_metric_tensor_dictionary:
    """
    # matrix 1

    # we want the name of the environment

    env_name = environment_name

    # extract relevant information
    data_env = data_dictionary[env_name]

    # making a strong assumption here that all experiments in this
    # environment will have the same number of steps, same number of tasks
    # same number of algorithms
    tasks = list(data_env.keys())
    algorithms = list(data_env[tasks[0]].keys())
    runs = list(data_env[tasks[0]][algorithms[0]].keys())
    steps = list(data_env[tasks[0]][algorithms[0]][runs[0]].keys())
    absolute_metrics = list(
        data_env[tasks[0]][algorithms[0]][runs[0]][steps[-1]].keys()
    )

    mean_absolute_metrics = [
        metric for metric in absolute_metrics if metric.split("_")[0].lower() == "mean"
    ]

    num_tasks = len(tasks)
    num_steps = len(steps)

    # create a dictionary of matrices
    metric_dictionary = {}

    for metric in mean_absolute_metrics:
        metric_dictionary[metric] = {}
        for algorithm in algorithms:
            metric_dictionary[metric][algorithm] = np.zeros(
                shape=(len(runs), len(tasks))
            )

    # now we need to populate the matrices
    for metric in mean_absolute_metrics:
        for algorithm in algorithms:
            for i, run in enumerate(runs):
                for j, task in enumerate(tasks):
                    metric_dictionary[metric][algorithm][i][j] = data_env[task][
                        algorithm
                    ][run][steps[-1]][metric]

    # now normalize the metrics that aren't already in range [0, 1]

    # we need to metrics that must be normalized
    non_norm_metrics = [f"mean_{metric}" for metric in metrics_to_normalize]

    for metric in non_norm_metrics:
        for task in range(num_tasks):
            min_max_array = np.array([])

            for algorithm in algorithms:
                min_max_array = np.concatenate(
                    (min_max_array, metric_dictionary[metric][algorithm][:, task])
                )
            min_ = np.min(min_max_array)
            max_ = np.max(min_max_array)

            for algorithm in algorithms:
                metric_dictionary[metric][algorithm][:, task] = (
                    metric_dictionary[metric][algorithm][:, task] - min_
                ) / (max_ - min_)

    metric_dictionary_return = metric_dictionary

    # next we create matrix 2

    # create master dictionary with all metrics
    master_metric_dictionary = {}

    for metric in mean_absolute_metrics:
        master_metric_dictionary[metric] = {}
        for algorithm in algorithms:
            master_metric_dictionary[metric][algorithm] = []

    for step in steps:

        metric_dictionary = {}
        for metric in mean_absolute_metrics:
            metric_dictionary[metric] = {}
            for algorithm in algorithms:
                metric_dictionary[metric][algorithm] = np.zeros(
                    shape=(len(runs), len(tasks))
                )

        # now we need to populate the matrices
        for metric in mean_absolute_metrics:
            for algorithm in algorithms:
                for i, run in enumerate(runs):
                    for j, task in enumerate(tasks):
                        metric_dictionary[metric][algorithm][i][j] = data_env[task][
                            algorithm
                        ][run][step][metric]

        for metric in mean_absolute_metrics:
            for algorithm in algorithms:
                master_metric_dictionary[metric][algorithm].append(
                    metric_dictionary[metric][algorithm]
                )

    final_metric_tensor_dictionary = {}
    for metric in mean_absolute_metrics:
        final_metric_tensor_dictionary[metric] = {}
        for algorithm in algorithms:

            final_metric_tensor_dictionary[metric][algorithm] = np.stack(
                master_metric_dictionary[metric][algorithm], axis=2
            )

    return metric_dictionary_return, final_metric_tensor_dictionary
