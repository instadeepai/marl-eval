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

import numpy as np

# Full dataset tests
matrix_1_expected_data = {
    "mean_norm_return": {
        "algo_1": np.array(
            [[0.40277778, 0.51388889, 0.76388889], [0.20833333, 0.33333333, 0.375]]
        ),
        "algo_2": np.array(
            [[0.36111111, 0.36111111, 0.43055556], [0.33333333, 0.43055556, 0.43055556]]
        ),
        "algo_3": np.array(
            [[0.43055556, 0.40277778, 0.76388889], [0.47222222, 0.58333333, 0.5]]
        ),
    },
    "mean_win_rate": {
        "algo_1": np.array([[0.2, 0.5, 0.7], [0.8, 0.7, 0.6]]),
        "algo_2": np.array([[0.6, 0.9, 0.3], [0.4, 0.8, 0.3]]),
        "algo_3": np.array([[0.1, 0.5, 0.2], [0.1, 0.2, 0.1]]),
    },
}

sample_efficiency_matrix_expected_data = {
    "mean_norm_return": {
        "algo_1": np.array(
            [
                [
                    [0.27777778, 0.16666667, 0.30555556],
                    [0.19444444, 0.38888889, 0.27777778],
                    [0.36111111, 0.91666667, 0.69444444],
                ],
                [
                    [0.13888889, 0.27777778, 0.38888889],
                    [0.5, 0.80555556, 0.61111111],
                    [0.08333333, 0.63888889, 0.63888889],
                ],
            ]
        ),
        "algo_2": np.array(
            [
                [
                    [0.08333333, 0.38888889, 0.5],
                    [0.38888889, 0.47222222, 0.36111111],
                    [0.02777778, 0.36111111, 0.91666667],
                ],
                [
                    [0.55555556, 0.25, 0.52777778],
                    [0.72222222, 0.5, 0.33333333],
                    [0.19444444, 0.16666667, 0.66666667],
                ],
            ]
        ),
        "algo_3": np.array(
            [
                [
                    [0.05555556, 0.11111111, 0.19444444],
                    [0.69444444, 0.58333333, 0.94444444],
                    [0.86111111, 0.41666667, 0.69444444],
                ],
                [
                    [0.72222222, 0.80555556, 0.25],
                    [0.72222222, 0.38888889, 0.41666667],
                    [0.66666667, 0.55555556, 0.63888889],
                ],
            ]
        ),
    },
    "mean_win_rate": {
        "algo_1": np.array(
            [
                [[0.8, 0.5, 0.4], [0.2, 0.5, 0.1], [0.2, 0.6, 0.8]],
                [
                    [0.7, 0.8, 0.4],
                    [0.5, 0.1, 0.2],
                    [0.4, 0.3, 0.2],
                ],
            ]
        ),
        "algo_2": np.array(
            [
                [[0.4, 0.7, 0.4], [0.9, 0.1, 0.1], [0.1, 0.3, 0.1]],
                [[0.2, 0.8, 0.1], [0.2, 0.8, 0.8], [0.2, 0.3, 0.5]],
            ]
        ),
        "algo_3": np.array(
            [
                [
                    [
                        0.1,
                        0.3,
                        0.5,
                    ],
                    [
                        0.6,
                        0.2,
                        0.2,
                    ],
                    [
                        0.3,
                        0.3,
                        0.2,
                    ],
                ],
                [[0.3, 0.2, 0.7], [0.1, 0.4, 0.5], [0.3, 0.3, 0.3]],
            ]
        ),
    },
}

# Single task test data

matrix_1_expected_data_single_task = {
    "mean_norm_return": {
        "algo_1": np.array([[0.40277778], [0.20833333]]),
        "algo_2": np.array([[0.36111111], [0.33333333]]),
        "algo_3": np.array([[0.43055556], [0.47222222]]),
    },
    "mean_win_rate": {
        "algo_1": np.array([[0.2], [0.8]]),
        "algo_2": np.array([[0.6], [0.4]]),
        "algo_3": np.array([[0.1], [0.1]]),
    },
}

sample_efficiency_matrix_expected_data_single_task = {
    "mean_norm_return": {
        "algo_1": np.array(
            [
                [[0.27777778, 0.16666667, 0.30555556]],
                [[0.13888889, 0.27777778, 0.38888889]],
            ]
        ),
        "algo_2": np.array(
            [[[0.08333333, 0.38888889, 0.5]], [[0.55555556, 0.25, 0.52777778]]]
        ),
        "algo_3": np.array(
            [[[0.05555556, 0.11111111, 0.19444444]], [[0.72222222, 0.80555556, 0.25]]]
        ),
    },
    "mean_win_rate": {
        "algo_1": np.array([[[0.8, 0.5, 0.4]], [[0.7, 0.8, 0.4]]]),
        "algo_2": np.array([[[0.4, 0.7, 0.4]], [[0.2, 0.8, 0.1]]]),
        "algo_3": np.array([[[0.1, 0.3, 0.5]], [[0.3, 0.2, 0.7]]]),
    },
}

# Single algorithm multiple task test data
matrix_1_expected_data_single_algorithm = {
    "mean_norm_return": {
        "algo_1": np.array(
            [[0.453125, 0.51388889, 0.76388889], [0.234375, 0.33333333, 0.375]]
        ),
    },
    "mean_win_rate": {
        "algo_1": np.array([[0.2, 0.5, 0.7], [0.8, 0.7, 0.6]]),
    },
}

sample_efficiency_matrix_expected_data_single_algorithm = {
    "mean_norm_return": {
        "algo_1": np.array(
            [
                [
                    [0.3125, 0.1875, 0.34375],
                    [0.19444444, 0.38888889, 0.27777778],
                    [0.36111111, 0.91666667, 0.69444444],
                ],
                [
                    [0.15625, 0.3125, 0.4375],
                    [0.5, 0.80555556, 0.61111111],
                    [0.08333333, 0.63888889, 0.63888889],
                ],
            ]
        ),
    },
    "mean_win_rate": {
        "algo_1": np.array(
            [
                [[0.8, 0.5, 0.4], [0.2, 0.5, 0.1], [0.2, 0.6, 0.8]],
                [[0.7, 0.8, 0.4], [0.5, 0.1, 0.2], [0.4, 0.3, 0.2]],
            ]
        ),
    },
}

expected_processed_data = {
    "env_1": {
        "task_1": {
            "algo_1": {
                "43289": {
                    "STEP_1": {
                        "step_count": 10006,
                        "return": [1, 2, 3, 4],
                        "win_rate": [0.8],
                        "mean_return": 2.5,
                        "norm_return": [
                            0.1111111111111111,
                            0.2222222222222222,
                            0.3333333333333333,
                            0.4444444444444444,
                        ],
                        "mean_norm_return": 0.2777777777777778,
                        "mean_win_rate": 0.8,
                    },
                    "STEP_2": {
                        "step_count": 20008,
                        "return": [1, 2, 1, 2],
                        "win_rate": [0.5],
                        "mean_return": 1.5,
                        "norm_return": [
                            0.1111111111111111,
                            0.2222222222222222,
                            0.1111111111111111,
                            0.2222222222222222,
                        ],
                        "mean_norm_return": 0.16666666666666666,
                        "mean_win_rate": 0.5,
                    },
                    "STEP_3": {
                        "step_count": 30000,
                        "return": [0, 1, 2, 8],
                        "win_rate": [0.4],
                        "mean_return": 2.75,
                        "norm_return": [
                            0.0,
                            0.1111111111111111,
                            0.2222222222222222,
                            0.8888888888888888,
                        ],
                        "mean_norm_return": 0.3055555555555555,
                        "mean_win_rate": 0.4,
                    },
                    "absolute_metrics": {
                        "return": [5, 8, 4, 1, 2, 3, 5, 1],
                        "win_rate": [0.2],
                        "mean_return": 3.625,
                        "norm_return": [
                            0.5555555555555556,
                            0.8888888888888888,
                            0.4444444444444444,
                            0.1111111111111111,
                            0.2222222222222222,
                            0.3333333333333333,
                            0.5555555555555556,
                            0.1111111111111111,
                        ],
                        "mean_norm_return": 0.4027777777777778,
                        "mean_win_rate": 0.2,
                    },
                },
                "42": {
                    "STEP_1": {
                        "step_count": 10006,
                        "return": [1, 1, 2, 1],
                        "win_rate": [0.7],
                        "mean_return": 1.25,
                        "norm_return": [
                            0.1111111111111111,
                            0.1111111111111111,
                            0.2222222222222222,
                            0.1111111111111111,
                        ],
                        "mean_norm_return": 0.1388888888888889,
                        "mean_win_rate": 0.7,
                    },
                    "STEP_2": {
                        "step_count": 20008,
                        "return": [1, 2, 5, 2],
                        "win_rate": [0.8],
                        "mean_return": 2.5,
                        "norm_return": [
                            0.1111111111111111,
                            0.2222222222222222,
                            0.5555555555555556,
                            0.2222222222222222,
                        ],
                        "mean_norm_return": 0.2777777777777778,
                        "mean_win_rate": 0.8,
                    },
                    "STEP_3": {
                        "step_count": 30000,
                        "return": [3, 1, 6, 4],
                        "win_rate": [0.4],
                        "mean_return": 3.5,
                        "norm_return": [
                            0.3333333333333333,
                            0.1111111111111111,
                            0.6666666666666666,
                            0.4444444444444444,
                        ],
                        "mean_norm_return": 0.3888888888888889,
                        "mean_win_rate": 0.4,
                    },
                    "absolute_metrics": {
                        "return": [1, 5, 1, 1, 0, 0, 5, 2],
                        "win_rate": [0.8],
                        "mean_return": 1.875,
                        "norm_return": [
                            0.1111111111111111,
                            0.5555555555555556,
                            0.1111111111111111,
                            0.1111111111111111,
                            0.0,
                            0.0,
                            0.5555555555555556,
                            0.2222222222222222,
                        ],
                        "mean_norm_return": 0.20833333333333334,
                        "mean_win_rate": 0.8,
                    },
                },
            },
            "algo_2": {
                "43289": {
                    "STEP_1": {
                        "step_count": 10006,
                        "return": [1, 1, 1, 0],
                        "win_rate": [0.4],
                        "mean_return": 0.75,
                        "norm_return": [
                            0.1111111111111111,
                            0.1111111111111111,
                            0.1111111111111111,
                            0.0,
                        ],
                        "mean_norm_return": 0.08333333333333333,
                        "mean_win_rate": 0.4,
                    },
                    "STEP_2": {
                        "step_count": 20008,
                        "return": [0, 5, 0, 9],
                        "win_rate": [0.7],
                        "mean_return": 3.5,
                        "norm_return": [0.0, 0.5555555555555556, 0.0, 1.0],
                        "mean_norm_return": 0.3888888888888889,
                        "mean_win_rate": 0.7,
                    },
                    "STEP_3": {
                        "step_count": 30000,
                        "return": [8, 5, 2, 3],
                        "win_rate": [0.4],
                        "mean_return": 4.5,
                        "norm_return": [
                            0.8888888888888888,
                            0.5555555555555556,
                            0.2222222222222222,
                            0.3333333333333333,
                        ],
                        "mean_norm_return": 0.49999999999999994,
                        "mean_win_rate": 0.4,
                    },
                    "absolute_metrics": {
                        "return": [1, 1, 8, 5, 1, 2, 0, 8],
                        "win_rate": [0.6],
                        "mean_return": 3.25,
                        "norm_return": [
                            0.1111111111111111,
                            0.1111111111111111,
                            0.8888888888888888,
                            0.5555555555555556,
                            0.1111111111111111,
                            0.2222222222222222,
                            0.0,
                            0.8888888888888888,
                        ],
                        "mean_norm_return": 0.36111111111111105,
                        "mean_win_rate": 0.6,
                    },
                },
                "42": {
                    "STEP_1": {
                        "step_count": 10006,
                        "return": [1, 5, 8, 6],
                        "win_rate": [0.2],
                        "mean_return": 5.0,
                        "norm_return": [
                            0.1111111111111111,
                            0.5555555555555556,
                            0.8888888888888888,
                            0.6666666666666666,
                        ],
                        "mean_norm_return": 0.5555555555555556,
                        "mean_win_rate": 0.2,
                    },
                    "STEP_2": {
                        "step_count": 20008,
                        "return": [2, 2, 2, 3],
                        "win_rate": [0.8],
                        "mean_return": 2.25,
                        "norm_return": [
                            0.2222222222222222,
                            0.2222222222222222,
                            0.2222222222222222,
                            0.3333333333333333,
                        ],
                        "mean_norm_return": 0.25,
                        "mean_win_rate": 0.8,
                    },
                    "STEP_3": {
                        "step_count": 30000,
                        "return": [1, 1, 8, 9],
                        "win_rate": [0.1],
                        "mean_return": 4.75,
                        "norm_return": [
                            0.1111111111111111,
                            0.1111111111111111,
                            0.8888888888888888,
                            1.0,
                        ],
                        "mean_norm_return": 0.5277777777777778,
                        "mean_win_rate": 0.1,
                    },
                    "absolute_metrics": {
                        "return": [1, 2, 1, 2, 2, 3, 8, 5],
                        "win_rate": [0.4],
                        "mean_return": 3.0,
                        "norm_return": [
                            0.1111111111111111,
                            0.2222222222222222,
                            0.1111111111111111,
                            0.2222222222222222,
                            0.2222222222222222,
                            0.3333333333333333,
                            0.8888888888888888,
                            0.5555555555555556,
                        ],
                        "mean_norm_return": 0.3333333333333333,
                        "mean_win_rate": 0.4,
                    },
                },
            },
            "algo_3": {
                "43289": {
                    "STEP_1": {
                        "step_count": 10006,
                        "return": [0, 0, 2, 0],
                        "win_rate": [0.1],
                        "mean_return": 0.5,
                        "norm_return": [0.0, 0.0, 0.2222222222222222, 0.0],
                        "mean_norm_return": 0.05555555555555555,
                        "mean_win_rate": 0.1,
                    },
                    "STEP_2": {
                        "step_count": 20008,
                        "return": [0, 1, 1, 2],
                        "win_rate": [0.3],
                        "mean_return": 1.0,
                        "norm_return": [
                            0.0,
                            0.1111111111111111,
                            0.1111111111111111,
                            0.2222222222222222,
                        ],
                        "mean_norm_return": 0.1111111111111111,
                        "mean_win_rate": 0.3,
                    },
                    "STEP_3": {
                        "step_count": 30000,
                        "return": [0, 0, 2, 5],
                        "win_rate": [0.5],
                        "mean_return": 1.75,
                        "norm_return": [
                            0.0,
                            0.0,
                            0.2222222222222222,
                            0.5555555555555556,
                        ],
                        "mean_norm_return": 0.19444444444444445,
                        "mean_win_rate": 0.5,
                    },
                    "absolute_metrics": {
                        "return": [1, 0, 2, 5, 0, 6, 9, 8],
                        "win_rate": [0.1],
                        "mean_return": 3.875,
                        "norm_return": [
                            0.1111111111111111,
                            0.0,
                            0.2222222222222222,
                            0.5555555555555556,
                            0.0,
                            0.6666666666666666,
                            1.0,
                            0.8888888888888888,
                        ],
                        "mean_norm_return": 0.4305555555555555,
                        "mean_win_rate": 0.1,
                    },
                },
                "42": {
                    "STEP_1": {
                        "step_count": 10006,
                        "return": [1, 8, 9, 8],
                        "win_rate": [0.3],
                        "mean_return": 6.5,
                        "norm_return": [
                            0.1111111111111111,
                            0.8888888888888888,
                            1.0,
                            0.8888888888888888,
                        ],
                        "mean_norm_return": 0.7222222222222222,
                        "mean_win_rate": 0.3,
                    },
                    "STEP_2": {
                        "step_count": 20008,
                        "return": [8, 9, 7, 5],
                        "win_rate": [0.2],
                        "mean_return": 7.25,
                        "norm_return": [
                            0.8888888888888888,
                            1.0,
                            0.7777777777777778,
                            0.5555555555555556,
                        ],
                        "mean_norm_return": 0.8055555555555556,
                        "mean_win_rate": 0.2,
                    },
                    "STEP_3": {
                        "step_count": 30000,
                        "return": [2, 1, 1, 5],
                        "win_rate": [0.7],
                        "mean_return": 2.25,
                        "norm_return": [
                            0.2222222222222222,
                            0.1111111111111111,
                            0.1111111111111111,
                            0.5555555555555556,
                        ],
                        "mean_norm_return": 0.25,
                        "mean_win_rate": 0.7,
                    },
                    "absolute_metrics": {
                        "return": [1, 6, 8, 9, 0, 5, 2, 3],
                        "win_rate": [0.1],
                        "mean_return": 4.25,
                        "norm_return": [
                            0.1111111111111111,
                            0.6666666666666666,
                            0.8888888888888888,
                            1.0,
                            0.0,
                            0.5555555555555556,
                            0.2222222222222222,
                            0.3333333333333333,
                        ],
                        "mean_norm_return": 0.4722222222222222,
                        "mean_win_rate": 0.1,
                    },
                },
            },
        },
        "task_2": {
            "algo_1": {
                "43289": {
                    "STEP_1": {
                        "step_count": 10006,
                        "return": [1, 2, 3, 1],
                        "win_rate": [0.2],
                        "mean_return": 1.75,
                        "norm_return": [
                            0.1111111111111111,
                            0.2222222222222222,
                            0.3333333333333333,
                            0.1111111111111111,
                        ],
                        "mean_norm_return": 0.19444444444444442,
                        "mean_win_rate": 0.2,
                    },
                    "STEP_2": {
                        "step_count": 20008,
                        "return": [1, 2, 5, 6],
                        "win_rate": [0.5],
                        "mean_return": 3.5,
                        "norm_return": [
                            0.1111111111111111,
                            0.2222222222222222,
                            0.5555555555555556,
                            0.6666666666666666,
                        ],
                        "mean_norm_return": 0.38888888888888884,
                        "mean_win_rate": 0.5,
                    },
                    "STEP_3": {
                        "step_count": 30000,
                        "return": [1, 2, 5, 2],
                        "win_rate": [0.1],
                        "mean_return": 2.5,
                        "norm_return": [
                            0.1111111111111111,
                            0.2222222222222222,
                            0.5555555555555556,
                            0.2222222222222222,
                        ],
                        "mean_norm_return": 0.2777777777777778,
                        "mean_win_rate": 0.1,
                    },
                    "absolute_metrics": {
                        "return": [1, 5, 8, 9, 4, 2, 8, 0],
                        "win_rate": [0.5],
                        "mean_return": 4.625,
                        "norm_return": [
                            0.1111111111111111,
                            0.5555555555555556,
                            0.8888888888888888,
                            1.0,
                            0.4444444444444444,
                            0.2222222222222222,
                            0.8888888888888888,
                            0.0,
                        ],
                        "mean_norm_return": 0.5138888888888888,
                        "mean_win_rate": 0.5,
                    },
                },
                "42": {
                    "STEP_1": {
                        "step_count": 10006,
                        "return": [0, 5, 6, 7],
                        "win_rate": [0.5],
                        "mean_return": 4.5,
                        "norm_return": [
                            0.0,
                            0.5555555555555556,
                            0.6666666666666666,
                            0.7777777777777778,
                        ],
                        "mean_norm_return": 0.5,
                        "mean_win_rate": 0.5,
                    },
                    "STEP_2": {
                        "step_count": 20008,
                        "return": [5, 8, 7, 9],
                        "win_rate": [0.1],
                        "mean_return": 7.25,
                        "norm_return": [
                            0.5555555555555556,
                            0.8888888888888888,
                            0.7777777777777778,
                            1.0,
                        ],
                        "mean_norm_return": 0.8055555555555556,
                        "mean_win_rate": 0.1,
                    },
                    "STEP_3": {
                        "step_count": 30000,
                        "return": [0, 8, 9, 5],
                        "win_rate": [0.2],
                        "mean_return": 5.5,
                        "norm_return": [
                            0.0,
                            0.8888888888888888,
                            1.0,
                            0.5555555555555556,
                        ],
                        "mean_norm_return": 0.6111111111111112,
                        "mean_win_rate": 0.2,
                    },
                    "absolute_metrics": {
                        "return": [1, 1, 5, 6, 1, 1, 1, 8],
                        "win_rate": [0.7],
                        "mean_return": 3.0,
                        "norm_return": [
                            0.1111111111111111,
                            0.1111111111111111,
                            0.5555555555555556,
                            0.6666666666666666,
                            0.1111111111111111,
                            0.1111111111111111,
                            0.1111111111111111,
                            0.8888888888888888,
                        ],
                        "mean_norm_return": 0.33333333333333337,
                        "mean_win_rate": 0.7,
                    },
                },
            },
            "algo_2": {
                "43289": {
                    "STEP_1": {
                        "step_count": 10006,
                        "return": [2, 5, 4, 3],
                        "win_rate": [0.9],
                        "mean_return": 3.5,
                        "norm_return": [
                            0.2222222222222222,
                            0.5555555555555556,
                            0.4444444444444444,
                            0.3333333333333333,
                        ],
                        "mean_norm_return": 0.3888888888888889,
                        "mean_win_rate": 0.9,
                    },
                    "STEP_2": {
                        "step_count": 20008,
                        "return": [1, 1, 6, 9],
                        "win_rate": [0.1],
                        "mean_return": 4.25,
                        "norm_return": [
                            0.1111111111111111,
                            0.1111111111111111,
                            0.6666666666666666,
                            1.0,
                        ],
                        "mean_norm_return": 0.4722222222222222,
                        "mean_win_rate": 0.1,
                    },
                    "STEP_3": {
                        "step_count": 30000,
                        "return": [3, 5, 4, 1],
                        "win_rate": [0.1],
                        "mean_return": 3.25,
                        "norm_return": [
                            0.3333333333333333,
                            0.5555555555555556,
                            0.4444444444444444,
                            0.1111111111111111,
                        ],
                        "mean_norm_return": 0.3611111111111111,
                        "mean_win_rate": 0.1,
                    },
                    "absolute_metrics": {
                        "return": [1, 1, 0, 1, 2, 5, 9, 7],
                        "win_rate": [0.9],
                        "mean_return": 3.25,
                        "norm_return": [
                            0.1111111111111111,
                            0.1111111111111111,
                            0.0,
                            0.1111111111111111,
                            0.2222222222222222,
                            0.5555555555555556,
                            1.0,
                            0.7777777777777778,
                        ],
                        "mean_norm_return": 0.3611111111111111,
                        "mean_win_rate": 0.9,
                    },
                },
                "42": {
                    "STEP_1": {
                        "step_count": 10006,
                        "return": [5, 6, 8, 7],
                        "win_rate": [0.2],
                        "mean_return": 6.5,
                        "norm_return": [
                            0.5555555555555556,
                            0.6666666666666666,
                            0.8888888888888888,
                            0.7777777777777778,
                        ],
                        "mean_norm_return": 0.7222222222222222,
                        "mean_win_rate": 0.2,
                    },
                    "STEP_2": {
                        "step_count": 20008,
                        "return": [1, 0, 8, 9],
                        "win_rate": [0.8],
                        "mean_return": 4.5,
                        "norm_return": [
                            0.1111111111111111,
                            0.0,
                            0.8888888888888888,
                            1.0,
                        ],
                        "mean_norm_return": 0.5,
                        "mean_win_rate": 0.8,
                    },
                    "STEP_3": {
                        "step_count": 30000,
                        "return": [1, 2, 5, 4],
                        "win_rate": [0.8],
                        "mean_return": 3.0,
                        "norm_return": [
                            0.1111111111111111,
                            0.2222222222222222,
                            0.5555555555555556,
                            0.4444444444444444,
                        ],
                        "mean_norm_return": 0.3333333333333333,
                        "mean_win_rate": 0.8,
                    },
                    "absolute_metrics": {
                        "return": [1, 0, 1, 0, 5, 8, 9, 7],
                        "win_rate": [0.8],
                        "mean_return": 3.875,
                        "norm_return": [
                            0.1111111111111111,
                            0.0,
                            0.1111111111111111,
                            0.0,
                            0.5555555555555556,
                            0.8888888888888888,
                            1.0,
                            0.7777777777777778,
                        ],
                        "mean_norm_return": 0.4305555555555556,
                        "mean_win_rate": 0.8,
                    },
                },
            },
            "algo_3": {
                "43289": {
                    "STEP_1": {
                        "step_count": 10006,
                        "return": [1, 8, 9, 7],
                        "win_rate": [0.6],
                        "mean_return": 6.25,
                        "norm_return": [
                            0.1111111111111111,
                            0.8888888888888888,
                            1.0,
                            0.7777777777777778,
                        ],
                        "mean_norm_return": 0.6944444444444444,
                        "mean_win_rate": 0.6,
                    },
                    "STEP_2": {
                        "step_count": 20008,
                        "return": [6, 5, 5, 5],
                        "win_rate": [0.2],
                        "mean_return": 5.25,
                        "norm_return": [
                            0.6666666666666666,
                            0.5555555555555556,
                            0.5555555555555556,
                            0.5555555555555556,
                        ],
                        "mean_norm_return": 0.5833333333333334,
                        "mean_win_rate": 0.2,
                    },
                    "STEP_3": {
                        "step_count": 30000,
                        "return": [9, 8, 9, 8],
                        "win_rate": [0.2],
                        "mean_return": 8.5,
                        "norm_return": [
                            1.0,
                            0.8888888888888888,
                            1.0,
                            0.8888888888888888,
                        ],
                        "mean_norm_return": 0.9444444444444444,
                        "mean_win_rate": 0.2,
                    },
                    "absolute_metrics": {
                        "return": [1, 1, 2, 4, 7, 7, 7, 0],
                        "win_rate": [0.5],
                        "mean_return": 3.625,
                        "norm_return": [
                            0.1111111111111111,
                            0.1111111111111111,
                            0.2222222222222222,
                            0.4444444444444444,
                            0.7777777777777778,
                            0.7777777777777778,
                            0.7777777777777778,
                            0.0,
                        ],
                        "mean_norm_return": 0.4027777777777778,
                        "mean_win_rate": 0.5,
                    },
                },
                "42": {
                    "STEP_1": {
                        "step_count": 10006,
                        "return": [1, 8, 8, 9],
                        "win_rate": [0.1],
                        "mean_return": 6.5,
                        "norm_return": [
                            0.1111111111111111,
                            0.8888888888888888,
                            0.8888888888888888,
                            1.0,
                        ],
                        "mean_norm_return": 0.7222222222222222,
                        "mean_win_rate": 0.1,
                    },
                    "STEP_2": {
                        "step_count": 20008,
                        "return": [8, 2, 2, 2],
                        "win_rate": [0.4],
                        "mean_return": 3.5,
                        "norm_return": [
                            0.8888888888888888,
                            0.2222222222222222,
                            0.2222222222222222,
                            0.2222222222222222,
                        ],
                        "mean_norm_return": 0.38888888888888895,
                        "mean_win_rate": 0.4,
                    },
                    "STEP_3": {
                        "step_count": 30000,
                        "return": [1, 4, 5, 5],
                        "win_rate": [0.5],
                        "mean_return": 3.75,
                        "norm_return": [
                            0.1111111111111111,
                            0.4444444444444444,
                            0.5555555555555556,
                            0.5555555555555556,
                        ],
                        "mean_norm_return": 0.4166666666666667,
                        "mean_win_rate": 0.5,
                    },
                    "absolute_metrics": {
                        "return": [0, 5, 6, 6, 8, 9, 8, 0],
                        "win_rate": [0.2],
                        "mean_return": 5.25,
                        "norm_return": [
                            0.0,
                            0.5555555555555556,
                            0.6666666666666666,
                            0.6666666666666666,
                            0.8888888888888888,
                            1.0,
                            0.8888888888888888,
                            0.0,
                        ],
                        "mean_norm_return": 0.5833333333333333,
                        "mean_win_rate": 0.2,
                    },
                },
            },
        },
        "task_3": {
            "algo_1": {
                "43289": {
                    "STEP_1": {
                        "step_count": 10006,
                        "return": [1, 1, 2, 9],
                        "win_rate": [0.2],
                        "mean_return": 3.25,
                        "norm_return": [
                            0.1111111111111111,
                            0.1111111111111111,
                            0.2222222222222222,
                            1.0,
                        ],
                        "mean_norm_return": 0.3611111111111111,
                        "mean_win_rate": 0.2,
                    },
                    "STEP_2": {
                        "step_count": 20008,
                        "return": [8, 8, 9, 8],
                        "win_rate": [0.6],
                        "mean_return": 8.25,
                        "norm_return": [
                            0.8888888888888888,
                            0.8888888888888888,
                            1.0,
                            0.8888888888888888,
                        ],
                        "mean_norm_return": 0.9166666666666666,
                        "mean_win_rate": 0.6,
                    },
                    "STEP_3": {
                        "step_count": 30000,
                        "return": [5, 5, 6, 9],
                        "win_rate": [0.8],
                        "mean_return": 6.25,
                        "norm_return": [
                            0.5555555555555556,
                            0.5555555555555556,
                            0.6666666666666666,
                            1.0,
                        ],
                        "mean_norm_return": 0.6944444444444444,
                        "mean_win_rate": 0.8,
                    },
                    "absolute_metrics": {
                        "return": [1, 2, 9, 9, 9, 9, 9, 7],
                        "win_rate": [0.7],
                        "mean_return": 6.875,
                        "norm_return": [
                            0.1111111111111111,
                            0.2222222222222222,
                            1.0,
                            1.0,
                            1.0,
                            1.0,
                            1.0,
                            0.7777777777777778,
                        ],
                        "mean_norm_return": 0.7638888888888888,
                        "mean_win_rate": 0.7,
                    },
                },
                "42": {
                    "STEP_1": {
                        "step_count": 10006,
                        "return": [0, 0, 1, 2],
                        "win_rate": [0.4],
                        "mean_return": 0.75,
                        "norm_return": [
                            0.0,
                            0.0,
                            0.1111111111111111,
                            0.2222222222222222,
                        ],
                        "mean_norm_return": 0.08333333333333333,
                        "mean_win_rate": 0.4,
                    },
                    "STEP_2": {
                        "step_count": 20008,
                        "return": [6, 6, 5, 6],
                        "win_rate": [0.3],
                        "mean_return": 5.75,
                        "norm_return": [
                            0.6666666666666666,
                            0.6666666666666666,
                            0.5555555555555556,
                            0.6666666666666666,
                        ],
                        "mean_norm_return": 0.6388888888888888,
                        "mean_win_rate": 0.3,
                    },
                    "STEP_3": {
                        "step_count": 30000,
                        "return": [7, 6, 6, 4],
                        "win_rate": [0.2],
                        "mean_return": 5.75,
                        "norm_return": [
                            0.7777777777777778,
                            0.6666666666666666,
                            0.6666666666666666,
                            0.4444444444444444,
                        ],
                        "mean_norm_return": 0.6388888888888888,
                        "mean_win_rate": 0.2,
                    },
                    "absolute_metrics": {
                        "return": [5, 5, 5, 4, 2, 3, 2, 1],
                        "win_rate": [0.6],
                        "mean_return": 3.375,
                        "norm_return": [
                            0.5555555555555556,
                            0.5555555555555556,
                            0.5555555555555556,
                            0.4444444444444444,
                            0.2222222222222222,
                            0.3333333333333333,
                            0.2222222222222222,
                            0.1111111111111111,
                        ],
                        "mean_norm_return": 0.375,
                        "mean_win_rate": 0.6,
                    },
                },
            },
            "algo_2": {
                "43289": {
                    "STEP_1": {
                        "step_count": 10006,
                        "return": [1, 0, 0, 0],
                        "win_rate": [0.1],
                        "mean_return": 0.25,
                        "norm_return": [0.1111111111111111, 0.0, 0.0, 0.0],
                        "mean_norm_return": 0.027777777777777776,
                        "mean_win_rate": 0.1,
                    },
                    "STEP_2": {
                        "step_count": 20008,
                        "return": [0, 2, 5, 6],
                        "win_rate": [0.3],
                        "mean_return": 3.25,
                        "norm_return": [
                            0.0,
                            0.2222222222222222,
                            0.5555555555555556,
                            0.6666666666666666,
                        ],
                        "mean_norm_return": 0.3611111111111111,
                        "mean_win_rate": 0.3,
                    },
                    "STEP_3": {
                        "step_count": 30000,
                        "return": [8, 8, 8, 9],
                        "win_rate": [0.1],
                        "mean_return": 8.25,
                        "norm_return": [
                            0.8888888888888888,
                            0.8888888888888888,
                            0.8888888888888888,
                            1.0,
                        ],
                        "mean_norm_return": 0.9166666666666666,
                        "mean_win_rate": 0.1,
                    },
                    "absolute_metrics": {
                        "return": [5, 6, 8, 9, 1, 2, 0, 0],
                        "win_rate": [0.1, 0.5],
                        "mean_return": 3.875,
                        "norm_return": [
                            0.5555555555555556,
                            0.6666666666666666,
                            0.8888888888888888,
                            1.0,
                            0.1111111111111111,
                            0.2222222222222222,
                            0.0,
                            0.0,
                        ],
                        "mean_norm_return": 0.4305555555555556,
                        "mean_win_rate": 0.3,
                    },
                },
                "42": {
                    "STEP_1": {
                        "step_count": 10006,
                        "return": [1, 2, 2, 2],
                        "win_rate": [0.2],
                        "mean_return": 1.75,
                        "norm_return": [
                            0.1111111111111111,
                            0.2222222222222222,
                            0.2222222222222222,
                            0.2222222222222222,
                        ],
                        "mean_norm_return": 0.19444444444444445,
                        "mean_win_rate": 0.2,
                    },
                    "STEP_2": {
                        "step_count": 20008,
                        "return": [0, 0, 0, 6],
                        "win_rate": [0.3],
                        "mean_return": 1.5,
                        "norm_return": [0.0, 0.0, 0.0, 0.6666666666666666],
                        "mean_norm_return": 0.16666666666666666,
                        "mean_win_rate": 0.3,
                    },
                    "STEP_3": {
                        "step_count": 30000,
                        "return": [0, 7, 8, 9],
                        "win_rate": [0.5],
                        "mean_return": 6.0,
                        "norm_return": [
                            0.0,
                            0.7777777777777778,
                            0.8888888888888888,
                            1.0,
                        ],
                        "mean_norm_return": 0.6666666666666666,
                        "mean_win_rate": 0.5,
                    },
                    "absolute_metrics": {
                        "return": [0, 0, 2, 3, 5, 4, 8, 9],
                        "win_rate": [0.3],
                        "mean_return": 3.875,
                        "norm_return": [
                            0.0,
                            0.0,
                            0.2222222222222222,
                            0.3333333333333333,
                            0.5555555555555556,
                            0.4444444444444444,
                            0.8888888888888888,
                            1.0,
                        ],
                        "mean_norm_return": 0.4305555555555556,
                        "mean_win_rate": 0.3,
                    },
                },
            },
            "algo_3": {
                "43289": {
                    "STEP_1": {
                        "step_count": 10006,
                        "return": [8, 9, 7, 7],
                        "win_rate": [0.3],
                        "mean_return": 7.75,
                        "norm_return": [
                            0.8888888888888888,
                            1.0,
                            0.7777777777777778,
                            0.7777777777777778,
                        ],
                        "mean_norm_return": 0.861111111111111,
                        "mean_win_rate": 0.3,
                    },
                    "STEP_2": {
                        "step_count": 20008,
                        "return": [1, 2, 5, 7],
                        "win_rate": [0.3],
                        "mean_return": 3.75,
                        "norm_return": [
                            0.1111111111111111,
                            0.2222222222222222,
                            0.5555555555555556,
                            0.7777777777777778,
                        ],
                        "mean_norm_return": 0.41666666666666663,
                        "mean_win_rate": 0.3,
                    },
                    "STEP_3": {
                        "step_count": 30000,
                        "return": [1, 8, 9, 7],
                        "win_rate": [0.2],
                        "mean_return": 6.25,
                        "norm_return": [
                            0.1111111111111111,
                            0.8888888888888888,
                            1.0,
                            0.7777777777777778,
                        ],
                        "mean_norm_return": 0.6944444444444444,
                        "mean_win_rate": 0.2,
                    },
                    "absolute_metrics": {
                        "return": [7, 8, 9, 7, 8, 9, 7, 0],
                        "win_rate": [0.2],
                        "mean_return": 6.875,
                        "norm_return": [
                            0.7777777777777778,
                            0.8888888888888888,
                            1.0,
                            0.7777777777777778,
                            0.8888888888888888,
                            1.0,
                            0.7777777777777778,
                            0.0,
                        ],
                        "mean_norm_return": 0.7638888888888888,
                        "mean_win_rate": 0.2,
                    },
                },
                "42": {
                    "STEP_1": {
                        "step_count": 10006,
                        "return": [5, 6, 6, 7],
                        "win_rate": [0.3],
                        "mean_return": 6.0,
                        "norm_return": [
                            0.5555555555555556,
                            0.6666666666666666,
                            0.6666666666666666,
                            0.7777777777777778,
                        ],
                        "mean_norm_return": 0.6666666666666666,
                        "mean_win_rate": 0.3,
                    },
                    "STEP_2": {
                        "step_count": 20008,
                        "return": [0, 5, 6, 9],
                        "win_rate": [0.3],
                        "mean_return": 5.0,
                        "norm_return": [
                            0.0,
                            0.5555555555555556,
                            0.6666666666666666,
                            1.0,
                        ],
                        "mean_norm_return": 0.5555555555555556,
                        "mean_win_rate": 0.3,
                    },
                    "STEP_3": {
                        "step_count": 30000,
                        "return": [1, 5, 8, 9],
                        "win_rate": [0.3],
                        "mean_return": 5.75,
                        "norm_return": [
                            0.1111111111111111,
                            0.5555555555555556,
                            0.8888888888888888,
                            1.0,
                        ],
                        "mean_norm_return": 0.6388888888888888,
                        "mean_win_rate": 0.3,
                    },
                    "absolute_metrics": {
                        "return": [8, 9, 7, 5, 2, 2, 1, 2],
                        "win_rate": [0.1],
                        "mean_return": 4.5,
                        "norm_return": [
                            0.8888888888888888,
                            1.0,
                            0.7777777777777778,
                            0.5555555555555556,
                            0.2222222222222222,
                            0.2222222222222222,
                            0.1111111111111111,
                            0.2222222222222222,
                        ],
                        "mean_norm_return": 0.5,
                        "mean_win_rate": 0.1,
                    },
                },
            },
        },
    },
    "extra": {
        "environment_list": {"env_1": ["task_1", "task_2", "task_3"]},
        "number_of_steps": 3,
        "number_of_runs": 2,
        "algorithm_list": ["algo_1", "algo_2", "algo_3"],
        "metric_list": {
            "env_1": [
                "return",
                "win_rate",
                "mean_return",
                "norm_return",
                "mean_norm_return",
                "mean_win_rate",
            ]
        },
        "evaluation_interval": {"env_1": 10000},
    },
}

expected_single_task_ci_data_returns = {
    "algo_1": {
        "mean": [0.20833333333333334, 0.2222222222222222, 0.3472222222222222],
        "ci": [0.04304211259673628, 0.03443369007738902, 0.025825267558041775],
    },
    "algo_2": {
        "mean": [0.3194444444444445, 0.3194444444444444, 0.5138888888888888],
        "ci": [0.14634318282890332, 0.04304211259673628, 0.008608422519347275],
    },
    "algo_3": {
        "mean": [0.3888888888888889, 0.45833333333333337, 0.2222222222222222],
        "ci": [0.20660214046433414, 0.21521056298368135, 0.017216845038694507],
    },
}

expected_single_task_ci_data_win_rates = {
    "algo_1": {
        "mean": [0.30000000000000004, 0.44999999999999996, 0.5],
        "ci": [0.061980642139300234, 0.09297096320895035, 0.18594192641790072],
    },
    "algo_2": {
        "mean": [0.15000000000000002, 0.3, 0.3],
        "ci": [0.030990321069650117, 0.0, 0.12396128427860047],
    },
    "algo_3": {"mean": [0.3, 0.3, 0.25], "ci": [0.0, 0.0, 0.030990321069650106]},
}
