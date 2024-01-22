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

import os
import neptune
from tqdm import tqdm
from typing import List
from colorama import Fore, Style


def pull_neptune_data(project_name: str, system_name: str, tag: List, store_directory: str ='.') -> None:
    """Pulls the experiments data from Neptune to local directory.
    
    Args:
        project_name (str): Name of the Neptune project.
        system_name (str): Name of the system (example: ff-ippo).
        tag (List): List of tags.
        store_directory (str, optional): Directory to store the data. Defaults current directory.
    """
    # Get the run ids
    project = neptune.init_project(project=project_name)
    runs_table_df = project.fetch_runs_table(state="inactive", tag=tag).to_pandas()
    runs_table_df = runs_table_df[
        runs_table_df["config/logger/system_name"] == system_name
    ]
    run_ids = runs_table_df["sys/id"].values.tolist()

    # Check if store_directory exists
    if not os.path.exists(store_directory):
        os.makedirs(store_directory)

    # Download the data
    for run_id in tqdm(run_ids, desc="Downloading Neptune Data"):
        run = neptune.init_run(
            project=project_name, with_id=run_id, mode="read-only"
        )
        data_key = list(run.get_structure()["metrics"].keys())[0]
        run[f"metrics/{data_key}"].download(
            destination=f"{store_directory}/{data_key}"
        )
        run.stop()
    
    print(f"{Fore.CYAN}{Style.BRIGHT}Data downloaded successfully!{Style.RESET_ALL}")
