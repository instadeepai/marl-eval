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
import zipfile
from typing import List

import neptune
from colorama import Fore, Style
from tqdm import tqdm


def pull_neptune_data(project_name: str, tag: List, store_directory: str = ".") -> None:
    """Pulls the experiments data from Neptune to local directory.

    Args:
        project_name (str): Name of the Neptune project.
        tag (List): List of tags.
        store_directory (str, optional): Directory to store the data.
    """
    # Get the run ids
    project = neptune.init_project(project=project_name)
    runs_table_df = project.fetch_runs_table(state="inactive", tag=tag).to_pandas()
    run_ids = runs_table_df["sys/id"].values.tolist()

    # Check if store_directory exists
    if not os.path.exists(store_directory):
        os.makedirs(store_directory)

    # Download and unzip the data
    itr = 0  # To create a unique directory for each unzipped file
    for run_id in tqdm(run_ids, desc="Downloading Neptune Data"):
        run = neptune.init_run(project=project_name, with_id=run_id, mode="read-only")
        for data_key in run.get_structure()["metrics"].keys():
            file_path = f"{store_directory}/{data_key}"
            run[f"metrics/{data_key}"].download(destination=file_path)
            try:
                with zipfile.ZipFile(file_path, "r") as zip_ref:
                    # Create a directory with to store unzipped data
                    os.makedirs(f"{store_directory}/{itr}", exist_ok=True)
                    # Unzip the data
                    zip_ref.extractall(f"{store_directory}/{itr}")
                    # Remove the zip file
                    os.remove(file_path)
            except zipfile.BadZipFile:
                # If it's not a zip file, just continue to the next file
                continue
            except Exception as e:
                print(f"An error occurred while unzipping or storing {file_path}: {e}")
            itr += 1
        run.stop()

    print(f"{Fore.CYAN}{Style.BRIGHT}Data downloaded successfully!{Style.RESET_ALL}")
