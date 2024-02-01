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

import copy
from typing import Any, Dict, List, Tuple, Union

import numpy as np
from colorama import Fore, Style

"""Tools for processing MARL experiment data."""


def lower_case_inputs(*args: Union[str, List[str]]) -> List:
    """Lower case all inputs.

    These inputs could be strings or lists of strings.
    """

    lower_case_data = [
        arg.lower() if isinstance(arg, str) else [a.lower() for a in arg]
        for arg in args
    ]

    # If lower cases data only contains one list, return the list.
    if len(lower_case_data) == 1 and isinstance(lower_case_data[0], list):
        return lower_case_data[0]

    return lower_case_data


def check_absolute_metric(steps: List) -> Union[str, None]:
    """Check that the absolute metric exist"""
    for step in steps:
        if "absolute" in step:
            return step
    return None


def check_comma_in_algo_names(
    algos: List, get_valid: bool = False
) -> Tuple[bool, list]:
    """Check that no algorithm names contain commas.

    Args:
        algos: list containing names of algorithms in experiment.
        get_valid: boolean dictating whether valid or invalid
            algorithm names should be returned. If `True` the names
            of valid algorithms will be returned and if `False` the
            names of invalid algorithms will be returned.
    """
    comma_in_names = [("," not in str(name)) for name in algos]
    names_valid = all(comma_in_names)

    if get_valid:
        names = [algos[i] for i in range(len(algos)) if comma_in_names[i] is True]
    else:
        names = [algos[i] for i in range(len(algos)) if comma_in_names[i] is False]

    return names_valid, names


def lower_case_dictionary_keys(
    dictionary: Dict[str, Dict[str, Any]]
) -> Dict[str, Dict[str, Any]]:
    """Recursively make all keys in a nested dictionary lower case."""

    new_dict = {}
    for key, value in dictionary.items():
        if isinstance(value, dict):
            new_dict[key.lower()] = lower_case_dictionary_keys(value)
        else:
            new_dict[key.lower()] = value
    return new_dict


def get_and_aggregate_data_single_task(
    processed_data: Dict[str, Any],
    metric_name: str,
    metrics_to_normalize: List[str],
    task_name: str,
    environment_name: str,
) -> Dict[str, Any]:
    """Compute the mean and 95% CI over all independent \
        experiment runs at each evaluation step for a given \
        environment and task.

    Args:
        processed_data: Dictionary containing processed data.
        metric_name: Name of metric to aggregate.
        metrics_to_normalize: List of metrics to normalize.
        task_name: Name of task to aggregate.
        environment_name: Name of environment to aggregate.
    """

    metrics_to_normalize, metric_name, task_name, environment_name = lower_case_inputs(
        metrics_to_normalize, metric_name, task_name, environment_name
    )

    if metric_name in metrics_to_normalize:
        metric_to_find = f"mean_norm_{metric_name}"
    else:
        metric_to_find = f"mean_{metric_name}"

    # Get the data for the given metric and environment
    task_data = processed_data[environment_name][task_name]

    # Get the algorithm names, number of runs and total steps
    algorithms = list(task_data.keys())
    runs = list(task_data[algorithms[0]].keys())
    steps = list(task_data[algorithms[0]][runs[0]].keys())

    # Remove absolute metric from steps.
    steps = [step for step in steps if "absolute" not in step.lower()]

    # Create a dictionary to store the mean and 95% CI for each algorithm
    mean_and_ci: Dict = {algorithm: {"mean": [], "ci": []} for algorithm in algorithms}

    for step in steps:
        # Loop over each algorithm
        for algorithm in algorithms:
            # Get the data for the given algorithm
            algorithm_data = task_data[algorithm]
            # Compute the mean and 95% CI for the given algorithm over all seeds
            # at a given step
            run_total = []
            for run in runs:
                run_total.append(algorithm_data[run][step][metric_to_find])

            mean_and_ci[algorithm]["mean"].append(np.mean(run_total))
            # Using central limit theorem to compute 95% CI
            mean_and_ci[algorithm]["ci"].append(1.96 * np.std(run_total) / np.sqrt(10))

    mean_and_ci["extra"] = processed_data["extra"]

    return mean_and_ci


def data_process_pipeline(  # noqa: C901
    raw_data: Dict[str, Dict[str, Any]],
    metrics_to_normalize: List[str],
) -> Dict[str, Dict[str, Any]]:
    """Function for processing raw input experiment data.

    Args:
        raw_data: Dictionary containing raw data that was read in
            from JSON file.
        metrics_to_normalize: A list of metric names for metrics that should
            be min/max normalised. These metric names should match the names as
            given in the raw dataset.

    Returns:
        processed_data: Dictionary containing processed experiment data where relevant
            metrics have been min/max normalised and the mean of all arrays in the
            dataset have been computed and added to the dataset.
    """

    metrics_to_normalize = lower_case_inputs(metrics_to_normalize)

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

        # Make all keys lower case
        raw_data = lower_case_dictionary_keys(raw_data)
        processed_data = copy.deepcopy(raw_data)

        metric_min_max_info: Dict[str, Any] = {}
        # Extra logs
        environment_list: Dict[str, Any] = {}
        algorithm_list = []
        metric_list: Dict[str, Any] = {}
        number_of_runs = 0
        number_of_steps = 0
        # Get the mean evaluation interval used in the experiment
        eval_interval: Dict[Any, Any] = {}

        for env, tasks in raw_data.items():
            environment_list[env] = []
            metric_list[env] = []
            eval_interval_per_env: list = []
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
                        step_count = 0
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
                                        ) / (
                                            metric_global_max - metric_global_min + 1e-6
                                        )
                                        processed_data[env][task][algorithm][run][step][
                                            f"norm_{metric}"
                                        ] = normed_metric_array.tolist()
                                        processed_data[env][task][algorithm][run][step][
                                            f"mean_norm_{metric}"
                                        ] = np.mean(normed_metric_array)
                                else:
                                    eval_interval_per_env.append(
                                        metrics[metric] - step_count
                                    )
                                    step_count = metrics[metric]
                            if metric_list[env] == []:
                                metric_list[env] = list(
                                    processed_data[env][task][algorithm][run][
                                        step
                                    ].keys()
                                )
                                if "step_count" in metric_list[env]:
                                    metric_list[env].remove("step_count")

                metric_min_max_info = {}
            eval_interval[env] = round(np.mean(eval_interval_per_env))

        processed_data["extra"] = {  # type: ignore
            "environment_list": environment_list,
            "number_of_steps": number_of_steps,
            "number_of_runs": number_of_runs,
            "algorithm_list": algorithm_list,
            "metric_list": metric_list,
            "evaluation_interval": eval_interval,
        }

        # Check that algorithm names do not contain commas
        algo_names_valid, invalid_algo_names = check_comma_in_algo_names(algorithm_list)
        if not algo_names_valid:
            raise ValueError(
                "Algorithm names must not contain commas."
                + f" Please update the following invalid names: {invalid_algo_names}\n"
            )

        return processed_data

    except Exception as e:
        print(e, ": There is an issue related to the format of the json file!")
        print(
            "We recommend using the DiagnoseData class from "
            + "marl_eval/utils/diagnose_data_errors.py to determine the error."
        )
        return raw_data


def create_matrices_for_rliable(  # noqa: C901
    data_dictionary: Dict[str, Dict[str, Any]],
    environment_name: str,
    metrics_to_normalize: List[str],
) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, Dict[str, Any]]]:
    """Creates two dictionaries containing arrays required for using the rliable tools.

        The first dictionary will have root keys corresponding to the metrics used
        in an experiment and subsequent keys corresponding to the Algorithms that were
        used in an experiment. For each metric algorithm pair a
        (number of runs x number of tasks) array is created containing as rows
        the normalised metric values across all tasks for a given independent
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
    environment_name, metrics_to_normalize = lower_case_inputs(
        environment_name, metrics_to_normalize
    )

    try:
        # Compute first arrays

        # Get the environment name
        env_name = environment_name

        # Extract relevant information
        data_env = data_dictionary[env_name]

        # Extract the extra params
        extra = data_dictionary.pop("extra")

        # Making a strong assumption here that all experiments in this
        # environment will have the same number of steps, same number of tasks
        # and same number of.
        tasks = list(data_env.keys())
        algorithms = list(data_env[tasks[0]].keys())
        runs = list(data_env[tasks[0]][algorithms[0]].keys())
        steps = list(data_env[tasks[0]][algorithms[0]][runs[0]].keys())

        # Check which step is the absolute metric
        absolute_metric_key = check_absolute_metric(steps)
        if absolute_metric_key is None:
            raise Exception(
                "The final logging step for\
            a given run should contain the absolute_metrics values\
            in a step called absolute_metrics."
            )

        absolute_metrics = list(
            data_env[tasks[0]][algorithms[0]][runs[0]][absolute_metric_key].keys()
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
                        # Get the metric data
                        metric_data = data_env[task][algorithm][run][
                            absolute_metric_key
                        ][metric]
                        # Compute the mean if it's a list, otherwise use as is
                        data = (
                            np.mean(metric_data)
                            if isinstance(metric_data, list)
                            else metric_data
                        )
                        # Store the data in the metric dictionary
                        metric_dictionary[metric][algorithm][i][j] = data

        metric_dictionary_return = metric_dictionary

        # Compute second arrays

        # Create master dictionary with all arrays
        master_metric_dictionary: Dict[str, Any] = {}

        for metric in mean_absolute_metrics:
            master_metric_dictionary[metric] = {}
            for algorithm in algorithms:
                master_metric_dictionary[metric][algorithm] = []

        # exclude the absolute metrics
        steps.remove(absolute_metric_key)
        for step in steps:
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
                            # Get the metric data
                            metric_data = data_env[task][algorithm][run][step][metric]
                            # Compute the mean if it's a list, otherwise use as is
                            data = (
                                np.mean(metric_data)
                                if isinstance(metric_data, list)
                                else metric_data
                            )
                            # Store the data in the metric dictionary
                            metric_dictionary[metric][algorithm][i][j] = data

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

        # Insert the extra info to the final metric tensor dict
        extra["evaluation_interval"] = extra["evaluation_interval"][env_name]
        final_metric_tensor_dictionary["extra"] = extra

        # Add extra back to data_dictionary
        data_dictionary["extra"] = extra

        return metric_dictionary_return, final_metric_tensor_dictionary

    except Exception as e:
        print(
            f"\n{Fore.RED}Unexpected error: {e}. There is an issue related to the "
            + "format of the json file!"
        )
        print(
            "We recommend using the DiagnoseData class from "
            + "`marl_eval/utils/diagnose_data_errors.py` for further "
            + f"investigation.\n{Style.RESET_ALL}"
        )
        raise
