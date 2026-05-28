import argparse
from pathlib import Path
import sys

CURRENT_DIR = Path(__file__).resolve().parent
SRCS_DIR = CURRENT_DIR.parent
if str(SRCS_DIR) not in sys.path:
    sys.path.insert(0, str(SRCS_DIR))

from utils import load_dataset, Datasets
from data import split_dataset

def main():
    parser = argparse.ArgumentParser(description="Split data into training and testing sets")
    parser.add_argument(
        "dataset_path",
        nargs="?",
        default="datasets/data_multilayer.csv",
        type=str,
        help="Path to the dataset file (e.g., dataset.csv)",
    )
    parser.add_argument(
        "train_size",
        nargs="?",
        default=0.8,
        type=float,
        help="Proportion of the dataset to include in the training set (e.g., 0.8 for 80%)",
    )
    parser.add_argument(
        "output_dir",
        nargs="?",
        default="datasets/split/",
        type=str,
        help="Path to the output directory for the split datasets",
    )
    try:
        split_dataset(Path(parser.parse_args().dataset_path), parser.parse_args().train_size, Path(parser.parse_args().output_dir))
    except Exception as exc:
        print(f"Error splitting dataset: {exc}")
        raise SystemExit(1)
    return 0

if __name__ == "__main__":
    main()