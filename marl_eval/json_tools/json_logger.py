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

import json
import os
import time
from typing import Dict, Optional


class JsonLogger:
    """Logger to create JSON files for reporting experiment results.

    This logger follows the suggested marl-eval protocol and was adapted
    from the implementation found in BenchMARL which can be viewed at:
    https://tinyurl.com/2t6fy548

    Args:
        path (str): folder path for saving the `metrics.json` file.
        algorithm_name (str): algorithm name e.g PPO.
        task_name (str): task name e.g 3s5z (for SMAC).
        environment_name (str): environment name e.g SMAC.
        seed (int): random seed of the experiment.
    """

    def __init__(
        self,
        path: str,
        algorithm_name: str,
        task_name: str,
        environment_name: str,
        seed: int,
    ):
        """Initialises the JsonLogger and creates a metrics file if it doesn't exist."""
        self.file_path = f"{path}/metrics.json"
        self.run_data: Dict = {"absolute_metrics": {}}

        # If the file already exists, load it
        if os.path.isfile(self.file_path):
            with open(self.file_path) as f:
                data = json.load(f)
        else:
            # Create the logging directory if it doesn't exist
            os.makedirs(path, exist_ok=True)
            data = {}

        # Merge the existing data with the new data
        self.data = data
        if environment_name not in self.data:
            self.data[environment_name] = {}
        if task_name not in self.data[environment_name]:
            self.data[environment_name][task_name] = {}
        if algorithm_name not in self.data[environment_name][task_name]:
            self.data[environment_name][task_name][algorithm_name] = {}
        self.data[environment_name][task_name][algorithm_name][
            f"seed_{seed}"
        ] = self.run_data

        with open(self.file_path, "w") as f:
            json.dump(self.data, f, indent=4)

    def write(
        self,
        timestep: int,
        key: str,
        value: float,
        evaluation_step: Optional[int] = None,
        is_absolute_metric: bool = False,
    ) -> None:
        """Writes a step to the json reporting file.

        Args:
            timestep (int): the current environment timestep.
            key (str): the name of the metric to be logged.
            value (float): the value of the metric to be logged.
            evaluation_step (int): the number of evaluations already run.
            is_absolute_metric (bool): whether the metric being logged is
                an absolute metric.
        """

        current_time = time.time()

        # This will ensure the first logged time is 0, which avoids taking compilation
        # into account for jax-based systems when plotting downstream.
        if evaluation_step == 0:
            self.start_time = current_time

        metrics: Dict = {key: [value]}

        if is_absolute_metric:
            self.run_data["absolute_metrics"].update(metrics)
        else:
            step_metrics = {  # type: ignore
                "step_count": timestep,
                "elapsed_time": current_time - self.start_time,
            } | metrics
            step_str = f"step_{evaluation_step}"
            if step_str in self.run_data:
                self.run_data[step_str].update(step_metrics)
            else:
                self.run_data[step_str] = step_metrics

        with open(self.file_path, "w") as f:
            json.dump(self.data, f, indent=4)
