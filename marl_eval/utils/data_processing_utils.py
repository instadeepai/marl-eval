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
import json
from typing import Any, Dict, List, Mapping, Optional, Tuple

import numpy as np

from marl_eval.utils.data_preprocessing_utils import data_preprocessing

"""Tools for processing MARL experiment data."""


def data_process_pipeline(  # noqa: C901
    raw_data: Mapping[str, Dict[str, Any]],
    metrics_to_normalize: List[str],
    reformat_json: Optional[bool] = False,
) -> Mapping[str, Dict[str, Any]]:
    """Function for processing raw input experiment data.

    Args:
        raw_data: Dictionary containing raw data that was read in
            from JSON file.
        metrics_to_normalize: A list of metric names for metrics that should
            be min/max normalised. These metric names should match the names as
            given in the raw dataset.
        reformat_json: check that the function was called for the second time
            after reformatting the json data.

    Returns:
        processed_data: Dictionary containing processed experiment data where relevant
            metrics have been min/max normalised and the mean of all arrays in the
            dataset have been computed and added to the dataset.
    """
    try:

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
                                    processed_data[env][task][algorithm][run][
                                        step
                                    ].keys()
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

    except Exception as e:
        if not reformat_json:
            print(e, ": There is an issue related to the format of the json file!")
            print("We will reformat the json data and recall the function")
            reformatted_data = data_preprocessing(raw_data)
            with open("./reformatted_data.json", "w+") as f:
                json.dump(reformatted_data, f, indent=4)
            data_process_pipeline(
                raw_data=reformatted_data,
                metrics_to_normalize=metrics_to_normalize,
                reformat_json=True,
            )
        else:
            print(
                e, ": You need to check the format that we provided of the json file."
            )

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
    try:
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

        def _select_metrics_for_plotting(absolute_metrics: list) -> list:
            """Select absolute metrics for plotting.

            Here only normalised versions of metrics that should be normalised
            should be chosen.
            """
            metrics_to_plot = []

            for metric in absolute_metrics:
                metric_split = metric.split("_")
                metric_in_absolute = len(
                    set(metric_split).intersection(set(metrics_to_normalize))
                )
                if metric.split("_")[0].lower() == "mean":
                    if metric_in_absolute > 0 and metric_split[1].lower() == "norm":
                        metrics_to_plot.append(metric)

                    elif metric_in_absolute == 0:
                        metrics_to_plot.append(metric)

            return metrics_to_plot

        mean_absolute_metrics = _select_metrics_for_plotting(absolute_metrics)

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

    except Exception as e:
        print(e, ": There is an issue related to the format of the json file!")
        print("We recommand using the data_preprocessing function to reformat the file")

    return metric_dictionary_return, final_metric_tensor_dictionary
