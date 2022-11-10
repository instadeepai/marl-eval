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
    raw_data: Mapping[str, Dict[str, Any]],
    metrics_to_normalize: List[str],
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

    def _compare_values(
        metric_min_max_info: Dict[str, Any], metric_values: list, metric: str
    ) -> None:
        """Compare list of metric values for a metric to the current global \
            min and max values for that metric.

        This is done in order to use the global min and max values downstream
        for normalising metrics.

        Args:
            metric_min_max_info: a dictionary containing global min and max
                values for all metrics.
            metric_values: a list containing metric data.
            metric: the name of the current metric for which global min and
                max data is being found.
        """

        min_per_step = np.min(metric_values)
        max_per_step = np.max(metric_values)
        if metric in list(metric_min_max_info.keys()):
            if metric_min_max_info[metric]["global_min"] > min_per_step:
                metric_min_max_info[metric]["global_min"] = min_per_step
            elif metric_min_max_info[metric]["global_max"] < max_per_step:
                metric_min_max_info[metric]["global_max"] = max_per_step
        else:
            metric_min_max_info[metric] = {
                "global_min": min_per_step,
                "global_max": max_per_step,
            }

    processed_data = copy.deepcopy(raw_data)
    metric_min_max_info: Dict[str, Any] = {}

    # Extra logs
    environment_list: Dict[str, Any] = {}
    algorithm_list = []
    metric_list: Dict[str, Any] = {}
    number_of_runs = 0
    number_of_steps = 0

    for env, tasks in raw_data.items():
        environment_list[env] = []
        metric_list[env] = []
        for task, algorithms in tasks.items():
            environment_list[env].append(task)
            for algorithm, runs in algorithms.items():
                if algorithm not in algorithm_list:
                    algorithm_list.append(algorithm)
                if number_of_runs == 0:
                    number_of_runs = len(runs.keys())
                for run, steps in runs.items():
                    if number_of_steps == 0:
                        number_of_steps = len(steps.keys()) - 1
                    for step, metrics in steps.items():
                        for metric in metrics_to_normalize:
                            # Find the global minimum and global maximum per task
                            _compare_values(
                                metric_min_max_info, metrics[metric], metric
                            )
            for algorithm, runs in algorithms.items():
                for run, steps in runs.items():
                    for step, metrics in steps.items():
                        for metric in metrics.keys():
                            if "step_count" not in metric:
                                # Mean
                                mean = np.mean(metrics[metric])
                                processed_data[env][task][algorithm][run][step][
                                    f"mean_{metric}"
                                ] = mean
                                if metric in metrics_to_normalize:
                                    # Normalization
                                    metric_array = np.array(metrics[metric])
                                    metric_global_min = metric_min_max_info[metric][
                                        "global_min"
                                    ]
                                    metric_global_max = metric_min_max_info[metric][
                                        "global_max"
                                    ]
                                    normed_metric_array = (
                                        metric_array - metric_global_min
                                    ) / (metric_global_max - metric_global_min)
                                    processed_data[env][task][algorithm][run][step][
                                        f"norm_{metric}"
                                    ] = normed_metric_array.tolist()
                                    processed_data[env][task][algorithm][run][step][
                                        f"mean_norm_{metric}"
                                    ] = np.mean(normed_metric_array)
                        if metric_list[env] == []:
                            metric_list[env] = list(
                                processed_data[env][task][algorithm][run][step].keys()
                            )
                            if "step_count" in metric_list[env]:
                                metric_list[env].remove("step_count")
            metric_min_max_info = {}

    processed_data["extra"] = {  # type: ignore
        "environment_list": environment_list,
        "number_of_steps": number_of_steps,
        "number_of_runs": number_of_runs,
        "algorithm_list": algorithm_list,
        "metric_list": metric_list,
    }
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


def create_matrices_for_rliable_second(  # noqa: C901
    data_dictionary: Mapping[str, Dict[str, Any]],
    environment_name: str,
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

    Returns:
        metric_dictionary_return: dictionary to be used by rliable tools
        final_metric_tensor_dictionary: dictionary to be used by rliable tools
    """
    # Compute first arrays

    # Get the extra data
    extra = data_dictionary["extra"]
    del data_dictionary["extra"]  # type: ignore

    # metric_dictionary_return
    metric_dictionary_return: Dict[str, Any] = {}

    for metric in extra["metric_list"][environment_name]:
        if "mean" in metric:
            metric_dictionary_return[metric] = {}
            for task, algorithms in data_dictionary[environment_name].items():
                for algorithm, runs in algorithms.items():
                    if algorithm not in metric_dictionary_return[metric].keys():
                        metric_dictionary_return[metric][algorithm] = []
                    aux = np.array([])
                    for run, steps in runs.items():
                        value = steps["ABSOLUTE_METRIC"][metric]
                        aux = np.append(aux, value)
                    metric_dictionary_return[metric][algorithm].append(aux)

            for algorithm in metric_dictionary_return[metric].keys():
                metric_dictionary_return[metric][algorithm] = np.array(
                    metric_dictionary_return[metric][algorithm]
                )
                metric_dictionary_return[metric][algorithm] = metric_dictionary_return[
                    metric
                ][algorithm].transpose()

    # master_metric_dictionary
    master_metric_dictionary: Dict[str, Any] = {}
    dimension = (
        extra["number_of_runs"],
        len(extra["environment_list"][environment_name]),
        extra["number_of_steps"],
    )
    for metric in extra["metric_list"][environment_name]:
        if "mean" in metric:
            master_metric_dictionary[metric] = {}
            for algorithms in data_dictionary[environment_name].values():
                for algorithm, runs in algorithms.items():
                    master_metric_dictionary[metric][algorithm] = np.zeros(dimension)
                    i = 0
                    for run, steps in runs.items():
                        j = 0
                        for task in extra["environment_list"][environment_name]:
                            p = 0
                            for step in steps.keys():
                                if "ABSOLUTE" not in step:
                                    master_metric_dictionary[metric][algorithm][i][j][
                                        p
                                    ] = data_dictionary[environment_name][task][
                                        algorithm
                                    ][
                                        run
                                    ][
                                        step
                                    ][
                                        metric
                                    ]
                                    p = p + 1
                            j = j + 1
                        i = i + 1
    return metric_dictionary_return, master_metric_dictionary
