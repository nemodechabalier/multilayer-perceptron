import argparse
import sys
from pathlib import Path

CURRENT_DIR = Path(__file__).resolve().parent
SRCS_DIR = CURRENT_DIR.parent
if str(SRCS_DIR) not in sys.path:
    sys.path.insert(0, str(SRCS_DIR))

from utils import load_json
from visualisation import compare

def main() -> int:
    """Entry point for generating Hogwarts house predictions."""

    parser = argparse.ArgumentParser(description="compare two models")
    parser.add_argument(
        "first_metric", 
        nargs="?", 
        help="Path to features, trained weights and describe file")
    parser.add_argument(
        "second_metric", 
        nargs="?", 
        help="Prediction CSV output path")
    args = parser.parse_args()

    try:
        payload1 = load_json(args.first_metric)
        payload2 = load_json(args.second_metric)
        compare(payload1, payload2)

        
    except Exception as e:
        print(f"Error: {e}")
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
