import argparse
from pathlib import Path
import sys
from pandas import DataFrame

CURRENT_DIR = Path(__file__).resolve().parent
SRCS_DIR = CURRENT_DIR.parent
if str(SRCS_DIR) not in sys.path:
    sys.path.insert(0, str(SRCS_DIR))

from visualisation import visualise_dataset
from utils import load_dataset, Datasets

def main():
    parser = argparse.ArgumentParser(description="Split data into training and testing sets")
    parser.add_argument(
        "dataset_path",
        nargs="?",
        default="datasets/data_multilayer.csv",
        type=str,
        help="Path to the dataset file (e.g., dataset.csv)",
    )
    try:
        df = load_dataset(Path(parser.parse_args().dataset_path), normalize=True)
        visualise_dataset(df)
    except Exception as exc:
        print(f"Error loading dataset: {exc}")
        raise SystemExit(1)
    return 0

if __name__ == "__main__":
    main()