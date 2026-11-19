import numpy as np
from math import sqrt, log
from pandas import DataFrame
from matplotlib import pyplot as plt
import json
from pathlib import Path


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

def visualisation(y_train, y_valid, title, show):
    Path("visu/train/").mkdir(parents=True, exist_ok=True)
    x = list(range(1, len(y_train) + 1))
    plt.figure()
    plt.plot(x, y_train, label = f'{title} train', color='red', linewidth=1.2, alpha=0.8)
    plt.plot(x, y_valid, label = f'{title} valid', color='blue', linewidth=1.2, alpha=0.8)
    plt.xlabel("Epochs")
    plt.ylabel(title)
    plt.legend()
    plt.grid(True)
    plt.savefig(f"visu/train/{title}")
    if show:
        plt.show()
