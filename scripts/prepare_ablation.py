import os
import subprocess
import sys

# Datasets for ablation study
DATASETS = [
    "d198", "d493", "u159", "gr96", "gr120", 
    "gr137", "bier127", "berlin52", "rat783", 
    "pr107", "pr152"
]

ALPHA = 7.0
EPOCHS = 2000
PYTHON_SOLVER = "../svrap_solver.py"
FORMATTED_DIR = "../formatted_dataset"

def train_model(dataset):
    dataset_path = os.path.join(FORMATTED_DIR, f"{dataset}.txt")
    if not os.path.exists(dataset_path):
        print(f"Skipping {dataset}: not found.")
        return

    print(f"Training {dataset} ({EPOCHS} epochs)...")
    cmd = [
        sys.executable, PYTHON_SOLVER,
        "--dataset", dataset_path,
        "--train",
        "--epochs", str(EPOCHS)
    ]
    subprocess.run(cmd, check=True)

if __name__ == "__main__":
    print(f"Preparing models for {len(DATASETS)} datasets...")
    for ds in DATASETS:
        try:
            train_model(ds)
        except Exception as e:
            print(f"Failed to train {ds}: {e}")
    print("All models prepared.")
