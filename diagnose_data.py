from marl_eval.utils.diagnose_data_errors import DiagnoseData
import json

data_dir = "./concatenated_json_files/metrics.json"

with open(data_dir, "r") as f:
    raw_data = json.load(f)

diagnose_obj = DiagnoseData(raw_data)

diagnosis_results = diagnose_obj.check_data()

for environment, results in diagnosis_results.items():
    print("Environment:", environment)
    print("Valid algorithms:", results["valid_algorithms"])
    print("Valid algorithm names:", results["valid_algorithm_names"])
    print("Valid runs:", results["valid_runs"])
    print("Valid steps:", results["valid_steps"])
    print("Valid metrics:", results["valid_metrics"])
    print()