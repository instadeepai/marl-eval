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

from typing import Any, Dict, Mapping
import copy

def check_algo(list_algo):
    if list_algo==[]:
        return True, []
    are_equal= True
    same_algos= sorted(list_algo[0])
    for i in range(1,len(list_algo)):
        if sorted(same_algos)!=sorted(list_algo[i]):
            are_equal=False
            same_algos= list(set(same_algos) & set(list_algo[i]))

    if not are_equal:
        print('The algorithms used accross the different tasks are not the same\n We will keep this list in the data:\n',sorted(same_algos))
    return are_equal, same_algos

def check_metric(list_metric):
    if list_metric==[]:
        return True, []
    are_equal= True
    same_metrics= sorted(list_metric[0])
    if "step_count" in same_metrics:
            same_metrics.remove("step_count")

    for i in range(1,len(list_metric)):
        if "step_count" in list_metric[i]:
            list_metric[i].remove("step_count")
        if sorted(same_metrics)!=sorted(list_metric[i]):
            are_equal=False
            same_metrics= list(set(same_metrics) & set(list_metric[i]))

    if not are_equal:
        print('The metrics used accross the different steps, runs, algorithms and scenarios are not the same\n We will keep this list in the data:\n',sorted(same_metrics))
    return are_equal, same_metrics

def check_runs(list_runs):
    if list_runs==[]:
        return True, []
    if list_runs.count(list_runs[0])==len(list_runs):
        return True, list_runs
    print("The number of runs is not identical through the different algorithms and scenarios.\n We will keep only "+ str(min(list_runs))+" runs.")
    return False, min(list_runs)

def check_steps(list_steps):
    if list_steps==[]:
        return True, []
    if list_steps.count(list_steps[0])==len(list_steps):
        return True, list_steps
    print("The number of steps is not identical through the different runs, algorithms and scenarios.\n We will keep only "+ str(min(list_steps))+" steps.")
    return False, min(list_steps)

def data_preprocessing(
    raw_data: Mapping[str, Dict[str, Any]],
)-> Mapping[str, Dict[str, Any]]:
    
    processed_data=copy.deepcopy(raw_data)
    list_algo=[]
    list_number_of_runs=[]
    list_number_of_steps=[]
    list_metrics=[]
    for env in raw_data.keys():
        for task in raw_data[env].keys():
            if task.lower()!=task:
                processed_data[env][task.lower()]=processed_data[env][task]
                del processed_data[env][task]
            for algorithm in raw_data[env][task].keys():
                if algorithm.upper()!=algorithm:
                    processed_data[env][task][algorithm.upper()]=processed_data[env][task][algorithm]
                    del processed_data[env][task][algorithm]
                list_number_of_runs.append(len(raw_data[env][task][algorithm].keys()))
                for run in raw_data[env][task][algorithm].keys():
                    list_number_of_steps.append(len(raw_data[env][task][algorithm][run].keys()))
                    for step in raw_data[env][task][algorithm][run].keys():
                        for metric in raw_data[env][task][algorithm][run][step].keys():
                            if metric.lower()!=metric:
                                processed_data[env][task][algorithm][run][step][metric.lower()]=processed_data[env][task][algorithm][run][step][metric]
                                del processed_data[env][task][algorithm][run][step][metric]
                        list_metrics.append(sorted(list(processed_data[env][task.lower()][algorithm.upper()][run][step].keys())))   
            list_algo.append(sorted(list(processed_data[env][task.lower()].keys())))
        is_same_algo, algos=check_algo(list_algo=list_algo)
        is_same_runs, runs_nb= check_runs(list_runs=list_number_of_runs)
        is_same_steps, steps_nb=check_steps(list_steps=list_number_of_steps)
        is_same_metric, metrics=check_metric(list_metric=list_metrics)


import json


# Read in and process data
METRICS_TO_NORMALIZE = ["return"]

with open("examples/example_results.json", "r") as f:
    raw_data = json.load(f)
data_preprocessing(raw_data)