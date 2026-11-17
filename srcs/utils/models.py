from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
from pandas import DataFrame

FeatureStats = Dict[str, Dict[str, float]]      # {"feature_1": {"mean": ..., "std": ...}}
Network = Dict[str, List[np.ndarray]]           # {"W": [...], "b": [...]}


@dataclass
class TrainState:
    config: Dict
    feature_cols: List[str]
    stats: FeatureStats

    df_train: DataFrame
    X_train: np.ndarray
    y_train: np.ndarray

    df_valid: DataFrame
    X_valid: np.ndarray
    y_valid: np.ndarray

    multilayer: Optional[Network] = None    # rempli après init_network
    history: Dict[str, List[float]] = field(
        default_factory=lambda: {"loss": [], "acc": [], "val_loss": [], "val_acc": []}
    )