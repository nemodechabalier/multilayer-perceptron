from dataclasses import dataclass
from pandas import DataFrame


@dataclass
class Datasets:
    raw: DataFrame
    train: DataFrame
    test: DataFrame