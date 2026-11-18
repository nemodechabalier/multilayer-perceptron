import numpy as np
from math import sqrt, log
from pandas import DataFrame
from matplotlib import pyplot as plt
import json
from pathlib import Path
from typing import Any
import sys
import copy

from .mathematical import apply_normalizer, forward, evaluate, backward
from .utils import save

from datetime import datetime
CURRENT_DIR = Path(__file__).resolve().parent
SRCS_DIR = CURRENT_DIR.parent
if str(SRCS_DIR) not in sys.path:
    sys.path.insert(0, str(SRCS_DIR))
from utils import TrainState



# def select_features(x_train, threshold):
#     """Return the indices of the columns to keep (computed on train only)."""
#     corr = np.abs(np.corrcoef(x_train, rowvar=False))
#     keep = []
#     for j in range(corr.shape[0]):
#         if all(corr[j, k] <= threshold for k in keep):
#             keep.append(j)
#     return keep

# def cleaning_datasets(datasets, config):
#     """Remove NaN rows, then drop features too correlated with one already kept."""

#     threshold = config.get("correlation")
#     if threshold is not None and not (0 <= threshold <= 1):
#         raise ValueError("'correlation' must be a number between 0 and 1.")
    
#     for key, x in datasets.items():
#         x_array = np.asarray(x)
#         if x_array.ndim != 2:
#             continue
#         datasets[key] = x_array[~np.isnan(x_array).any(axis=1)]

    
    
#     if threshold is not None:
#         if "x_train" not in datasets:
#             raise ValueError("Missing 'x_train' dataset for feature selection.")
#         keep = select_features(datasets["x_train"], threshold)

#         for key in ("x_train", "x_valid"):
#             if key not in datasets:
#                 continue

#             datasets[key] = datasets[key][:, keep]
#             print(f"Kept {len(keep)} features out of {datasets[key].shape[1]} for dataset '{key}' based on correlation threshold {threshold}.")
#         config["kept_features"] = keep
#     return datasets



def fit_normalizer(df, feature_cols=None):
    """Fit sur train: calcule mean/std et renvoie train normalise + stats."""
    norm_df = df.copy()

    if feature_cols is None:
        feature_cols = [c for c in norm_df.columns if c.startswith("feature_")]

    stats = {}
    for col in feature_cols:
        values = norm_df[col].astype(np.float64)
        mean = float(values.mean())
        std = float(values.std(ddof=0))

        if not np.isfinite(std) or std < 1e-12:
            std = 1.0

        norm_df[col] = (values - mean) / (std + 1e-8)
        stats[col] = {"mean": mean, "std": std}

    return norm_df, stats, feature_cols

def init_network(sizes, seed=1) -> dict :
    rng = np.random.default_rng(seed)   # créé une seule fois
    weights, biases = [], []
    for n_in, n_out in zip(sizes[:-1], sizes[1:]):
        print(n_in)
        limit = sqrt(6 / n_in)
        weights.append(rng.uniform(-limit, limit, (n_in, n_out)))
        biases.append(np.zeros(n_out))
    return {"W": weights, "b": biases}

def visualisation(y_train, y_valid, title):
    x = list(range(1, len(y_train) + 1))
    plt.figure()
    plt.plot(x, y_train, label = f'{title} train', color='red', linewidth=1.2, alpha=0.8)
    plt.plot(x, y_valid, label = f'{title} valid', color='blue', linewidth=1.2, alpha=0.8)
    plt.xlabel("Epochs")
    plt.ylabel(title)
    plt.legend()
    plt.grid(True)
    plt.savefig(f"visu/train/{title}")
    plt.show()

def mini_batch(state: TrainState):
    alpha = state.config.get("learning_rate")
    num_epochs = state.config.get("epochs")
    batch_size = state.config.get("batch_size")
    rng = np.random.default_rng(state.config.get("seed"))
    m = len(state.y_train)

    history = {"loss": [], "acc": [], "val_loss": [], "val_acc": []}
    early_stop = {"multilayer" : copy.deepcopy(state.multilayer), "epoch" : 0, "val_loss" : float("inf"), "last_epoch": 0}
    for epoch in range(num_epochs):
        indices = rng.permutation(m)
        X_shuffled = state.X_train[indices]
        y_shuffled = state.y_train[indices]
        for start_idx in range(0, m, batch_size):
            end_idx = min(start_idx + batch_size, m)
            X_batch = X_shuffled[start_idx:end_idx]
            y_batch = y_shuffled[start_idx:end_idx]
            n_layers = len(state.multilayer["W"])
            acts = forward(X_batch, state.multilayer)
            dW, db = backward(acts, y_batch, state.multilayer, n_layers)
            for l, (W, b) in enumerate(zip(state.multilayer["W"], state.multilayer["b"])):
                W -= alpha * dW[l] ; b -= alpha * db[l]
        stats = evaluate(state)
        for key, value in zip(history, (stats["loss"], stats["acc"], stats["val_loss"], stats["val_acc"])):
            history[key].append(value)
        print(f"epoch {epoch + 1:02d}/{num_epochs} - loss: {stats["loss"]:.4f} - val_loss: {stats["val_loss"]:.4f}"
              f" - acc: {stats["acc"]:.4f} - val_acc: {stats["val_acc"]:.4f} - precision: {stats["precision"]:.4f}"
              f" - rappel: {stats["rappel"]:.4f} - F1_score: {stats["F1_score"]:.4f}")
        if stats["val_loss"] < early_stop["val_loss"] :
            early_stop = {"multilayer" : copy.deepcopy(state.multilayer), "epoch" : epoch + 1, "val_loss" : stats["val_loss"], "last_epoch" : 0}
        early_stop["last_epoch"] += 1
        if early_stop["last_epoch"] > state.config.get("patience"):
            break

    return state.multilayer, history, early_stop


def train_models(state: TrainState):
    # clean_datasets = cleaning_datasets(datasets, config)
    print(f"Training models with config: {state.config}")

    norm_df, norm_stats, feature_cols = fit_normalizer(state.df_train)
    norm_df_valid = apply_normalizer(state.df_valid, norm_stats, feature_cols)

    diag = (norm_df["diagnosis"] == "M").astype(np.float64).to_numpy()
    state.y_train = np.column_stack((diag ,1 - diag))
    state.X_train = norm_df[feature_cols].to_numpy(dtype=np.float64)
    diag_valid = (norm_df_valid["diagnosis"] == "M").astype(np.float64).to_numpy()
    state.y_valid = np.column_stack((diag_valid ,1 - diag_valid))
    state.X_valid = norm_df_valid[feature_cols].to_numpy(dtype=np.float64)
    input_dim = state.X_train.shape[1]
    hidden_dims = state.config.get("layer")
    output_dim = 2
    sizes = [input_dim] + hidden_dims + [output_dim]
    state.multilayer = init_network(sizes, state.config.get("seed"))
    state.multilayer, history, early_stop = mini_batch(state)
    if early_stop["epoch"] < state.config.get("epochs"):
        print(f"Early stopping epoch {early_stop["epoch"]} with val_loss {early_stop["val_loss"]}")
        state.multilayer = early_stop["multilayer"]
    visualisation(history["acc"], history["val_acc"], "Accuracy")
    visualisation(history["loss"], history["val_loss"], "Loss")
    save(state.multilayer, norm_stats, "datasets/model.json",feature_cols)
    payload = {"config": state.config, "history": history}
    layers = [input_dim] + hidden_dims + [output_dim]
    dir_hist = state.config.get("path")
    Path(dir_hist).mkdir(parents=True, exist_ok=True)
    dir_hist = dir_hist + f"metriques_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.json"
    with open(dir_hist, "w", encoding="utf-8") as file_obj:
            json.dump(payload, file_obj, indent=2)


