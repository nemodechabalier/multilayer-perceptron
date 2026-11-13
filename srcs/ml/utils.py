import numpy as np
from math import sqrt, log
from pandas import DataFrame
from matplotlib import pyplot as plt
import json


def save(multilayer, norm_stats, output_path, features):
    payload = {
        "features": features,
        "weights": [W.tolist() for W in multilayer["W"]],
        "bias": [b.tolist() for b in multilayer["b"]],
        "mean": [norm_stats[col]["mean"] for col in features],
        "std": [norm_stats[col]["std"] for col in features],
    }
    with open(output_path, "w", encoding="utf-8") as file_obj:
        json.dump(payload, file_obj, indent=2)

