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
from utils import load_dataset, TrainState
# from data import split_dataset
import random
import sys

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
  if not isinstance(config, dict):
    raise ValueError("Config must be a dictionary.")

  missing = [k for k in KEYS if k not in config]
  if missing:
    raise ValueError(f"Missing config key(s): {', '.join(missing)}")


  layers = config["layer"]
  if (
      not isinstance(layers, list)
      or len(layers) < 2
      or not all(type(n) is int and n > 0 for n in layers)
  ):
    raise ValueError("'layer' must be a list of at least 2 positive integers.")

  epoch = config["epochs"]
  if type(epoch) is not int or epoch <= 20:
    raise ValueError("'epochs' must be an integer greater than 20.")

  batch_size = config["batch_size"]
  if type(batch_size) is not int or batch_size <= 0:
    raise ValueError("'batch_size' must be a positive integer.")

  lr = config["learning_rate"]
  if type(lr) not in (int, float) or lr <= 0:
    raise ValueError("'learning_rate' must be a positive number.")

  cor = config["correlation"]
  if type(cor) not in (int, float) or not (0 <= cor <= 1):
    raise ValueError("'correlation' must be a number between 0 and 1.")

  seed = config["seed"]
  if seed == "random":
    config["seed"] = random.randint(1, sys.maxsize)
  elif type(seed) is not int or seed < 0:
    raise ValueError("'seed' must be a non-negative integer or 'random'.")

  path = config["path"]
  if not isinstance(path, str) or not path.strip():
    raise ValueError("'path' must be a non-empty string.")
  if not path.endswith("/"):
    config["path"] = path + "/"

  patience = config["patience"]
  if type(patience) is not int or patience < 1:
    raise ValueError("'patience' must be an integer greater than 0.")

  optimizer = config["optimizer"]
  if not isinstance(optimizer, str) or optimizer not in ("adam", "sgd"):
    raise ValueError("'optimizer' must be 'adam' or 'sgd'.")

  early_stopping = config["early_stopping"]
  if not isinstance(early_stopping, bool):
    raise ValueError("'early_stopping' must be 'yes' or 'no'.")

  show = config["show"]
  if not isinstance(show, bool):
    raise ValueError("'show' must be 'yes' or 'no'.")
  return config


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--dataset_train',
        type=str,
        nargs='?',
        default='datasets/split/train.csv',
        help='Path to the training dataset train CSV file'
    )
    parser.add_argument(
        '--dataset_test',
        type=str,
        nargs='?',
        default='datasets/split/test.csv',
        help='Path to the training dataset teat CSV file'
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
        state = TrainState
        state.df_train = load_dataset(args.dataset_train, harmonize=True)
        state.df_valid = load_dataset(args.dataset_test, harmonize=True)
        state.config = load_config(args.config)
        validate_config(state.config)
        train_models(state)
    except Exception as exc:
        parser.error(str(exc))

if __name__ == "__main__":
    main()