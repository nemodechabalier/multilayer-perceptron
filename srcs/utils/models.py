from dataclasses import dataclass
from pandas import DataFrame
from typing import Dict, List
FeatureStats = Dict[str, Dict[str, float]]

@dataclass
class Datasets:
    df: DataFrame
    stats: FeatureStats

