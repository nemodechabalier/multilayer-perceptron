import argparse
from pathlib import Path
import sys
from sklearn.model_selection import train_test_split


CURRENT_DIR = Path(__file__).resolve().parent
SRCS_DIR = CURRENT_DIR.parent
if str(SRCS_DIR) not in sys.path:
    sys.path.insert(0, str(SRCS_DIR))

from utils import load_dataset, Datasets, save_dataset

def split_dataset(dataset_path: Path, train_size: float, output_dir: Path) -> Datasets:
    """Split the dataset into training and testing sets."""
    if train_size <= 0.1 or train_size >= 0.9:
        raise ValueError("train_size must be a float between 0.1 and 0.9.")
    try:
        df = load_dataset(dataset_path, normalize=False)
        if df is None:
            raise ValueError("Failed to load dataset.")
    except Exception as exc:
        print(f"Error loading dataset: {exc}")
        raise SystemExit(1)
    try:
        label_col = df.columns[1]
        train, test = train_test_split(df,
                                                 train_size=train_size,
                                                 stratify=df[label_col],
                                                 random_state=42)
        train = train.reset_index(drop=True)
        test = test.reset_index(drop=True)
        print(f"Dataset split into training set of shape {train.shape} and testing set of shape {test.shape}")
    except Exception as exc:
        print(f"Error splitting dataset: {exc}")
        raise SystemExit(1)
    print(test.head())
    print(f"Saving split datasets to {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    save_dataset(train, output_dir / "train.csv")
    save_dataset(test, output_dir / "test.csv")
    return train, test
    
    