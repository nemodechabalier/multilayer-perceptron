import csv
from pandas import DataFrame
from typing import Dict, List

def save_dataset(predictions: DataFrame, output_path: str) -> None:
    predictions.to_csv(output_path, index=False)