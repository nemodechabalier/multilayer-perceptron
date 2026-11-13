import numpy as np
from math import sqrt, log
from pandas import DataFrame
from matplotlib import pyplot as plt
from .utils import save
from .mathematical import apply_normalizer, forward, evaluate, backward
import json

def predict(df, data):
    feature_cols = data["features"]
    stats = {}
    for l, col in enumerate(feature_cols):
        stats[col] = {"mean": data["mean"][l], "std": data["std"][l]}
    norm_df = apply_normalizer(df, stats, feature_cols)
    diag = (norm_df["diagnosis"] == "M").astype(np.float64).to_numpy()
    y = np.column_stack((diag ,1 - diag))
    X = norm_df[feature_cols].to_numpy(dtype=np.float64)

    multilayer = {
        "W": [np.array(W, dtype=np.float64) for W in data["weights"]],
        "b": [np.array(b, dtype=np.float64) for b in data["bias"]],
    }
    P = forward(X, multilayer)