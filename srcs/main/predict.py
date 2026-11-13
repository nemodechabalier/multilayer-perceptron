import argparse
import sys
from pathlib import Path

CURRENT_DIR = Path(__file__).resolve().parent
SRCS_DIR = CURRENT_DIR.parent
if str(SRCS_DIR) not in sys.path:
    sys.path.insert(0, str(SRCS_DIR))

from utils import load_dataset, Datasets, load_json
from ml import predict


def main() -> int:
    """Entry point for generating Hogwarts house predictions."""

    parser = argparse.ArgumentParser(description="One-vs-all logistic regression prediction entrypoint.")
    parser.add_argument(
        "datasets", 
        nargs="?", 
        default="datasets/split/test.csv")
    parser.add_argument(
        "logreg_weights", 
        nargs="?", 
        default="datasets/model.json", 
        help="Path to features, trained weights and describe file")
    parser.add_argument(
        "--output", 
        nargs="?", 
        default="houses.csv", 
        help="Prediction CSV output path")
    args = parser.parse_args()

    try:

        df = load_dataset(args.datasets, harmonize=True)
        print(df)
        if df is None:
            print("Unable to load dataset.")
            return 1
        data = load_json(args.logreg_weights)
        predict(df, data)
    except Exception as e:
        print(f"Error: {e}")
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
