from pathlib import Path
from generate_degraded import generate_degraded_dataset
from restore import restore_dataset
from evaluate import evaluate_dataset

def main():
    project_root = Path(__file__).resolve().parents[1]
    dataset_dir = project_root / "dataset"

    print("Step 1/3: Generate degraded images")
    generate_degraded_dataset(dataset_dir)

    print("\nStep 2/3: Restore degraded images")
    restore_dataset(dataset_dir)

    print("\nStep 3/3: Evaluate results")
    evaluate_dataset(dataset_dir)

    print("\nDone.")
    print(f"Check output folder: {dataset_dir}")

if __name__ == "__main__":
    main()
