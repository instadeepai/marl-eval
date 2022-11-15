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

"""Tools to check the json file format and edit it."""

import copy
from typing import Any, Dict, List, Mapping


class DiagnoseData:
    """Class to diagnose the errors and"""

    def __init__(self, raw_data: Mapping[str, Dict[str, Any]]) -> None:
        """Init"""
        self.raw_data = raw_data

    def check_algo(self, list_algo: List) -> tuple:
        """Check that through the scenarios, the data share the same algorithms"""
        if list_algo == []:
            return True, []
        identical = True
        same_algos = sorted(list_algo[0])

        for i in range(1, len(list_algo)):
            if sorted(same_algos) != sorted(list_algo[i]):
                identical = False
                same_algos = list(set(same_algos) & set(list_algo[i]))

        if not identical:
            print(
                "The algorithms used accross the different tasks are not the same\n\
                We will keep this list in the data:\n",
                sorted(same_algos),
            )

        return identical, same_algos

    def check_metric(self, list_metric: List) -> tuple:
        """Check that through the steps, runs, algo and scenarios, the data share \
            the same list of metrics"""
        if list_metric == []:
            return True, []
        identical = True
        same_metrics = sorted(list_metric[0])

        if "step_count" in same_metrics:
            same_metrics.remove("step_count")

        for i in range(1, len(list_metric)):
            if "step_count" in list_metric[i]:
                list_metric[i].remove("step_count")
            if sorted(same_metrics) != sorted(list_metric[i]):
                identical = False
                same_metrics = list(set(same_metrics) & set(list_metric[i]))

        if not identical:
            print(
                "The metrics used accross the different steps, runs, algorithms\
                    and scenarios are not the same\n\
                    We will keep this list in the data:\n",
                sorted(same_metrics),
            )

        return identical, same_metrics

    def check_runs(self, list_nb_runs: List) -> tuple:
        """Check that through the algos, and tasks, the data share the same nb of run"""
        if list_nb_runs == []:
            return True, []

        if list_nb_runs.count(list_nb_runs[0]) == len(list_nb_runs):
            return True, list_nb_runs[0]

        print(
            "The number of runs is not identical through the different algorithms and scenarios.\n\
                We will keep only "
            + str(min(list_nb_runs))
            + " runs."
        )
        return False, min(list_nb_runs)

    def check_steps(self, list_nb_steps: List) -> tuple:
        """Check that through the different runs, algo and scenarios, \
            the data share the same number of steps"""
        if list_nb_steps == []:
            return True, []

        if list_nb_steps.count(list_nb_steps[0]) == len(list_nb_steps):
            return True, list_nb_steps[0]

        print(
            "The number of steps is not identical through the different runs, \
                algorithms and scenarios.\n We will keep only "
            + str(min(list_nb_steps))
            + " steps."
        )
        return False, min(list_nb_steps)

    def diagnose_error(self) -> Mapping[str, Dict[str, Any]]:  # noqa: C901
        """Check if there is any issue related to the format of the json file"""

        processed_data = copy.deepcopy(self.raw_data)

        for env in self.raw_data.keys():

            # List of algorithms used in the experiment across the tasks
            algorithms_used = []
            # List of nb or runs used across the algos and the tasks
            runs_used = []
            # List of nb of steps used across the runs, the algos and the tasks
            steps_used = []
            # List of metrics used across the steps, the runs, the algos and the tasks
            metrics_used = []

            for task in self.raw_data[env].keys():
                # Low-case the task names
                if task.lower() != task:
                    processed_data[env][task.lower()] = copy.deepcopy(
                        self.raw_data[env][task]
                    )
                    del processed_data[env][task]

                # Append the list of used algorithms across the tasks
                algorithms_used.append(
                    sorted(list(processed_data[env][task.lower()].keys()))
                )

                for algorithm in self.raw_data[env][task].keys():
                    # Up-case the algorithm name
                    if algorithm.upper() != algorithm:
                        processed_data[env][task.lower()][
                            algorithm.upper()
                        ] = copy.deepcopy(self.raw_data[env][task][algorithm])
                        del processed_data[env][task.lower()][algorithm]

                    # Append the number of runs used across the different algos
                    runs_used.append(
                        len(processed_data[env][task.lower()][algorithm.upper()].keys())
                    )

                    for run in self.raw_data[env][task][algorithm].keys():
                        # Append the number of steps used across the different runs.
                        steps_used.append(
                            len(
                                processed_data[env][task.lower()][algorithm.upper()][
                                    run
                                ].keys()
                            )
                        )

                        for step in self.raw_data[env][task][algorithm][run].keys():
                            for metric in self.raw_data[env][task][algorithm][run][
                                step
                            ].keys():
                                # Low-case the metric names
                                if metric.lower() != metric:
                                    processed_data[env][task.lower()][
                                        algorithm.upper()
                                    ][run][step][metric.lower()] = copy.deepcopy(
                                        self.raw_data[env][task][algorithm][run][step][
                                            metric
                                        ]
                                    )
                                    del processed_data[env][task.lower()][
                                        algorithm.upper()
                                    ][run][step][metric]

                            # Append the metrics names used across the different steps..
                            metrics_used.append(
                                sorted(
                                    list(
                                        processed_data[env][task.lower()][
                                            algorithm.upper()
                                        ][run][step].keys()
                                    )
                                )
                            )

                            # Up-case absolute metric
                            if "absolute" in step.lower() and step != "ABSOLUTE_METRIC":
                                processed_data[env][task.lower()][algorithm.upper()][
                                    run
                                ]["ABSOLUTE_METRIC"] = copy.deepcopy(
                                    self.raw_data[env][task][algorithm][run][step]
                                )
                                del processed_data[env][task.lower()][
                                    algorithm.upper()
                                ][run][step]

            # Check that the format don't issued any issue while using the tools
            valid_algo, _ = self.check_algo(list_algo=algorithms_used)
            valid_runs, _ = self.check_runs(list_nb_runs=runs_used)
            valid_steps, _ = self.check_steps(list_nb_steps=steps_used)
            valid_metric, _ = self.check_metric(list_metric=metrics_used)

            # Check that we have valid json file
            if valid_algo and valid_runs and valid_steps and valid_metric:
                print("Valid format for the task " + task + "!")

        return processed_data