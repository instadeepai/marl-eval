# marl-eval


## Data Structure for Raw Experiment data

In order to use the tools we suggest effectively, raw data JSON files are required to have the following structure :

```json
{
    "environment_name" : {
        "task_name" : {
            "algorithm_name": {
                "run_1": {
                    "step_1" : {
                        "step_count": <int>,
                        "metric_1": [<number_evaluation_episodes>],
                        "metric_2": [<number_evaluation_episodes>],
                    }
                    .
                    .
                    .
                    "step_k" : {
                        "step_count": <int>,
                        "metric_1": [<number_evaluation_episodes>],
                        "metric_2": [<number_evaluation_episodes>],
                    }
                    "absolute_metrics": {
                        "metric_1": [<number_evaluation_episodes>*10],
                        "metric_2": [<number_evaluation_episodes>*10]
                    }

                }
                .
                .
                .
                "run_n": {
                    "step_1" : {
                        "step_count": <int>,
                        "metric_1": [<number_evaluation_episodes>],
                        "metric_2": [<number_evaluation_episodes>],
                    }
                    .
                    .
                    .
                    "step_k" : {
                        "step_count": <int>,
                        "metric_1": [<number_evaluation_episodes>],
                        "metric_2": [<number_evaluation_episodes>],
                    }
                    "absolute_metrics": {
                        "metric_1": [<number_evaluation_episodes>*10],
                        "metric_2": [<number_evaluation_episodes>*10]
                    }
                }
            }
        }
    }
}
```
Here `run_1` to `run_n` correspond to the number of independent runs in a given experiment and `step_1` to `step_k` correspond to the number of logging steps in a given environment. `step_count` corresponds to the amount of steps taken by agents in the environment when logging occurs and the values logged for each relevant metric for a given logging step should be a list containing either 1 element for a metric such as a win rate which gets computed over multiple episodes or as many elements as evaluation episodes that we run at the logging step. The final logging step for a given run should contain the `absolute_metric` values for the given metric in an experiment with these list containing either 1 element or 10 times as many elements as evaluation episodes at each logging step.
