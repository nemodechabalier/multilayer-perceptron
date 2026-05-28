import pandas as pd
from pandas import DataFrame
from typing import Optional

def load_dataset(path: str) -> Optional[DataFrame]:
    """Load a CSV file and display its dimensions."""
    try:
        dataset = pd.read_csv(path)
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