# -----------------------------------------------------------------------------
#  This Python script is the primary entry point called by our judge. It runs
#  your code to generate anomaly scores, then evaluates those scores to produce
#  the final results.
# -----------------------------------------------------------------------------

import subprocess

# Step 1: Generate anomaly scores
subprocess.run(["./run.sh"], check=True)

# Step 2: Evaluate the generated scores
subprocess.run(
    [
        "python3",
        "evaluation/eval_main.py",
        "--device",
        "0",
        "--data_path",
        "./data/",
        "--dataset_name",
        "rayan_dataset",
        "--class_name",
        "all",
        "--output_dir",
        "./output",
        "--output_scores_dir",
        "./output_scores",
        "--save_csv",
        "True",
        "--save_json",
        "True",
        "--class_name_mapping_dir",
        "./evaluation/class_name_mapping.json",
    ],
    check=True,
)
