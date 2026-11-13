import pandas as pd
from pandas import DataFrame
from typing import Optional
import json


DATASET_COLUMNS = ["sample_id", "diagnosis"] + [f"feature_{index}" for index in range(1, 31)]

def load_json(path: str):
    """Load and return JSON data from a file path."""

    with open(path) as f:
        data = json.load(f)
    return data

def _harmonize_dataset(dataset: DataFrame) -> DataFrame:
    """Attach generic names and handle both raw and already-saved CSV files."""
    if len(dataset.columns) != len(DATASET_COLUMNS):
        return dataset

    dataset = dataset.copy()
    first_row = dataset.iloc[0].astype(str).str.lower().tolist()
    if first_row[:2] == ["sample_id", "diagnosis"] or first_row[:2] == ["id", "diagnosis"]:
        dataset = dataset.iloc[1:].reset_index(drop=True)

    dataset.columns = DATASET_COLUMNS

    for column in DATASET_COLUMNS:
        if column != "diagnosis":
            dataset[column] = pd.to_numeric(dataset[column], errors="coerce")

    return dataset

def load_dataset(path: str, harmonize: bool = False) -> Optional[DataFrame]:
    """Load a CSV file and display its dimensions."""
    try:
        dataset = pd.read_csv(path, header=None)
        if harmonize:
            dataset = _harmonize_dataset(dataset)
        print(f"Loading dataset of dimensions {dataset.shape}")
        return dataset
    except FileNotFoundError:
        print(f"Error: File '{path}' not found.")
        return None
    except pd.errors.EmptyDataError:
        print(f"Error: File '{path}' is empty.")
        return None
    except pd.errors.ParserError:
        print(f"Error: File '{path}' has invalid format.")
        return None
    except Exception as exc:
        print(f"Error: {exc}")
        return None