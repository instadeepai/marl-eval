# JSON tooling usage guide

## JSON logger

The JSON logger will write experiment data to JSON files in the format required for downstream aggregation and plotting with the MARL-eval tools. To initialise the logger the following arguments are required:

* `path`: the path where a file called `metrics.json` will be stored which will contain all logged metrics for a given experiment. Data will be stored in `<path>/metrics.json` by default. If a JSON file already exists at a particular path, new experiment data will be appended to it. MARL-eval currently does not support asynchronous logging. So if you intend to run distributed experiments, please create a unique `path` per experiment and concatenate all generated JSON files after all experiments have been run with the provided `concatenate_json_files` function.
* `algorithm_name`: the name of the algorithm being run in the current experiment.
* `task_name`: the name of the task in the current experiment.
* `environment_name`: the name of the environment in the current experiment.
* `seed`: the integer value of the seed used for pseudo-randomness in the current experiment.

An example of initialising the JSON logger could look something like:

```python
from marl_eval.json_tools import JsonLogger

json_logger = JsonLogger(
    path="experiment_results",
    algorithm_name="IPPO",
    task_name="2s3z",
    environment_name="SMAX",
    seed=42,
)
```

To write data to the logger, the `write` method takes in the following arguments:

* `timestep`: the current environment timestep at the time of evaluation.
* `key`: the name of the metric to be logged.
* `value`: the scalar value to be logged for the current metric.
* `evaluation_step`: the number of evaluations that have been performed so far.
* `is_absolute_metric`: a boolean flag indicating whether an absolute metric is being logged.

Suppose a the `4`th evaluation is being performed at environment timestep `40000` for the `episode_return` metric with a value of `12.9` then the `write` method could be used as follows:

```python
json_logger.write(
    timestep=40_000,
    key="episode_return",
    value=12.9,
    evaluation_step=4,
    is_absolute_metric=False,
)
```

In the case where the absolute metric for the `win_rate` metric with a value of `85.3` is logged at the `200`th evaluation after `2_000_000` timesteps, the `write` method would be called as follows:

```python
json_logger.write(
    timestep=2_000_000,
    key="win_rate",
    value=85.3,
    evaluation_step=200,
    is_absolute_metric=True,
)
```

## Neptune data pulling script
The `pull_neptune_data` script will download JSON data for multiple experiment runs from Neptune given a list of one or more Neptune experiment tags. The function accepts the following arguments:

* `project_name`: the name of the neptune project where data has been logged given as `<workspace_name>/<project_name>`.
* `tag`: a list of Neptune experiment tags for which JSON data should be downloaded.
* `store_directory`: a local directory where downloaded JSON files should be stored.
* `neptune_data_key`: a key in a particular Neptune run where JSON data has been stored. By default this while be `metrics` implying that the JSON file will be stored as `metrics/<metric_file_name>.zip` in a given Neptune run. For an example of how data is uploaded please see [here](https://github.com/instadeepai/Mava/blob/ce9a161a0b293549b2a34cd9a8d794ba7e0c9949/mava/utils/logger.py#L182).

In order to download data, the tool can be used as follows:

```python
from marl_eval.json_tools import pull_netpune_data

pull_netpune_data(
    project_name="DemoWorkspace/demo_project",
    tag=["experiment_1"],
    store_directory="./neptune_json_data",
)
```

## JSON file merging script
The `concatenate_json_files` function will merge all JSON files found in a given directory into a single JSON file ready to be used for downstream aggregation and plotting with MARL-eval. The function accepts the following arguments:

* `input_directory`: the path to the directory containing multiple JSON files. This directory can contain JSON files in arbitrarily nested directories.
* `output_json_path`: the path where the merged JSON file should be stored.

The function can be used as follows:

```python
from marl_eval.json_tools import concatenate_json_files

concatenate_json_files(
    input_directory="path/to/some/folder/",
    output_json_path="path/to/merged_file/folder/",
)
```
