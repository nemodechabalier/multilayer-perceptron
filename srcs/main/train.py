import argparse
from pathlib import Path
import sys
from typing import Any
import yaml

CURRENT_DIR = Path(__file__).resolve().parent
SRCS_DIR = CURRENT_DIR.parent
if str(SRCS_DIR) not in sys.path:
    sys.path.insert(0, str(SRCS_DIR))
from ml import train_models
from utils import load_dataset, Datasets
from data import split_dataset

KEYS = ("layer", "epochs", "batch_size", "learning_rate", "correlation")

def load_config(path: str) -> dict:
    try:
        with open(path, encoding="utf-8") as f:
            data = yaml.safe_load(f)
    except (OSError, yaml.YAMLError) as exc:
        raise ValueError(f"Cannot read config {path}: {exc}") from exc
    if not isinstance(data, dict):
        raise ValueError("Config must be a YAML mapping.")
    return data

def validate_config(config: dict) -> dict:
    missing = [k for k in KEYS if k not in config]
    if missing:
        raise ValueError(f"Missing config key(s): {', '.join(missing)}")

    layers = config["layer"]
    if (not isinstance(layers, list) or len(layers) < 2
            or not all(type(n) is int and n > 0 for n in layers)):
        raise ValueError("'layer' must be a list of at least 2 positive integers.")

    for key in ("epochs", "batch_size"):
        if type(config[key]) is not int or config[key] <= 0:
            raise ValueError(f"'{key}' must be a positive integer.")

    lr = config["learning_rate"]
    if type(lr) not in (int, float) or lr <= 0:
        raise ValueError("'learning_rate' must be a positive number.")

    lr = config["correlation"]
    if type(lr) not in (int, float) or not (0 <= lr <= 1):
        raise ValueError("'correlation' must be a number between 0 and 1.")

    return {k: config[k] for k in KEYS}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--dataset',
        type=str,
        nargs='?',
        default='datasets/split/train.csv',
        help='Path to the training dataset CSV file'
    )
    parser.add_argument(
        '--config',
        type=str,
        nargs='?',
        default='config/default.yaml',
        help='Path to the configuration YAML file'
    )
    args = parser.parse_args()
    try:
        dataset = load_dataset(args.dataset, harmonize=True)
        config = load_config(args.config)
        validate_config(config)
        train_models(dataset, config)
    except Exception as exc:
        parser.error(str(exc))

if __name__ == "__main__":
    main()