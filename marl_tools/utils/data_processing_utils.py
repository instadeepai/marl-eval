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
from typing import Any, Dict, List, Mapping, Tuple

import numpy as np

"""Tools for processing MARL experiment data."""


def data_process_pipeline(  # noqa: C901
    raw_data: Mapping[str, Dict[str, Any]], metrics_to_normalize: List[str]
) -> Mapping[str, Dict[str, Any]]:
    """Function for processing raw input experiment data.

    Args:
        raw_data: Dictionary containing raw data that was read in
            from JSON file.
        metrics_to_normalize: A list of metric names for metrics that should
            be min/max normalised. These metric names should match the names as
            given in the raw dataset

    Returns:
        processed_data: Dictionary containing processed experiment data where relevant
            metrics have been min/max normalised and the mean of all arrays in the
            dataset have been computed and added to the dataset.
    """

    metric_min_max_info: Dict[str, Any] = {}

    # Create global max, min dictionary
    for metric in metrics_to_normalize:
        for env in raw_data.keys():
            metric_min_max_info[env] = {}
            for scenario in raw_data[env]:
                metric_min_max_info[env][scenario] = {}
                metric_min_max_info[env][scenario][metric] = {}
                metric_min_max_info[env][scenario][metric]["global_min"] = 1_000_000
                metric_min_max_info[env][scenario][metric]["global_max"] = -1_000_000

    # Now we have to traverse the data to find the global min and max
    for metric in metrics_to_normalize:
        for env in raw_data.keys():
            for scenario in raw_data[env]:
                for algorithm in raw_data[env][scenario]:
                    for run in raw_data[env][scenario][algorithm]:
                        # exclude final step since it contains the absolute metrics
                        for step in list(
                            raw_data[env][scenario][algorithm][run].keys()
                        ):
                            min_val = np.min(
                                raw_data[env][scenario][algorithm][run][step][metric]
                            )
                            max_val = np.max(
                                raw_data[env][scenario][algorithm][run][step][metric]
                            )
                            if (
                                min_val
                                < metric_min_max_info[env][scenario][metric][
                                    "global_min"
                                ]
                            ):
                                metric_min_max_info[env][scenario][metric][
                                    "global_min"
                                ] = min_val
                            if (
                                max_val
                                > metric_min_max_info[env][scenario][metric][
                                    "global_max"
                                ]
                            ):
                                metric_min_max_info[env][scenario][metric][
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
                                    ][metric]["global_min"]
                                    metric_global_max = metric_min_max_info[env][
                                        scenario
                                    ][metric]["global_max"]
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


def create_matrices_for_rliable(  # noqa: C901
    data_dictionary: Mapping[str, Dict[str, Any]],
    environment_name: str,
    metrics_to_normalize: List[str],
) -> Tuple[Mapping[str, Dict[str, Any]], Mapping[str, Dict[str, Any]]]:
    """Creates two dictionaries containing arrays required for using the rliable tools.

        The first dictionary will have root keys corresponding to the metrics used
        in an experiment and subsequent keys corresponding to the Algorithms that were
        used in an experiment. For each metric algorithm pair a
        (number of runs x number of tasks) array is created containing as rows
        the normalised metric values acrossall tasks for a given independent
        experiment run.

        The second dictionary will have root keys corresponding to the metrics used
        in an experiment and subsequent keys corresponding to the Algorithms that
        were used in an experiment, but for each metric algorithm pair a
        (number of runs x number of tasks x number of logging steps) array is created
        where the rows correspond to the normalised metric values across
        all tasks for a given logging step of an independent experiment run.
        This dictionary will be used to produce the sample efficiency curves.


    Args:
        data_dictionary: Dictionary of data that has been processed using the
            data_process_pipeline function.
        environment_name: Name of environment for which arrays should be
            computed.
        metrics_to_normalize: List of metric names of metrics that should be
            normalised.

    Returns:
        metric_dictionary_return: dictionary to be used by rliable tools
        final_metric_tensor_dictionary: dictionary to be used by rliable tools
    """
    # Compute first arrays

    # Get the environment name
    env_name = environment_name

    # Extract relevant information
    data_env = data_dictionary[env_name]

    # Making a strong assumption here that all experiments in this
    # environment will have the same number of steps, same number of tasks
    # and same number of.
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

    # Create a dictionary of matrices with correct shape
    metric_dictionary: Dict[str, Any] = {}

    for metric in mean_absolute_metrics:
        metric_dictionary[metric] = {}
        for algorithm in algorithms:
            metric_dictionary[metric][algorithm] = np.zeros(
                shape=(len(runs), len(tasks))
            )

    # Populate the matrices
    for metric in mean_absolute_metrics:
        for algorithm in algorithms:
            for i, run in enumerate(runs):
                for j, task in enumerate(tasks):
                    metric_dictionary[metric][algorithm][i][j] = data_env[task][
                        algorithm
                    ][run][steps[-1]][metric]

    # Normalize the metrics that aren't already in range [0, 1]

    # Get metrics that must be normalized
    non_norm_metrics = [f"mean_{metric}" for metric in metrics_to_normalize]

    for metric in non_norm_metrics:
        for task in range(len(tasks)):  # type: ignore
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

    # Compute second arrays

    # Create master dictionary with all arrays
    master_metric_dictionary: Dict[str, Any] = {}

    for metric in mean_absolute_metrics:
        master_metric_dictionary[metric] = {}
        for algorithm in algorithms:
            master_metric_dictionary[metric][algorithm] = []

    # exclude the absolute metrics
    for step in steps[:-1]:

        metric_dictionary = {}
        for metric in mean_absolute_metrics:
            metric_dictionary[metric] = {}
            for algorithm in algorithms:
                metric_dictionary[metric][algorithm] = np.zeros(
                    shape=(len(runs), len(tasks))
                )

        # Now populate the matrices
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

    final_metric_tensor_dictionary: Dict[str, Any] = {}
    for metric in mean_absolute_metrics:
        final_metric_tensor_dictionary[metric] = {}
        for algorithm in algorithms:

            final_metric_tensor_dictionary[metric][algorithm] = np.stack(
                master_metric_dictionary[metric][algorithm], axis=2
            )

    return metric_dictionary_return, final_metric_tensor_dictionary
