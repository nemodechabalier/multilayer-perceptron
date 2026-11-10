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
    parser.add_argument(
        "feature_selection",
        nargs="*",
        type=str,
        default=["feature_1", "feature_2", "feature_3", "feature_4", "feature_5", "feature_6", "feature_7", "feature_8", "feature_9", "feature_10", "feature_11", "feature_12", "feature_13", "feature_14", "feature_15", "feature_16", "feature_17", "feature_18", "feature_19", "feature_20", "feature_21", "feature_22", "feature_23", "feature_24", "feature_25", "feature_26", "feature_27", "feature_28", "feature_29", "feature_30"],
        help="List of features to select for visualisation",
    )

    args = parser.parse_args()

    try:
        df = load_dataset(Path(args.dataset_path), harmonize=True)
        print(df.head())
        visualise_dataset(df, features=args.feature_selection)
    except Exception as exc:
        print(f"Error loading dataset: {exc}")
        raise SystemExit(1)
    return 0

if __name__ == "__main__":
    main()