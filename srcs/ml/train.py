import numpy as np
from math import sqrt, log
from pandas import DataFrame


def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def sigmoid_deriv(z):
    s = sigmoid(z)
    return s * (1 - s)


def softmax(z):
    exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))
    return exp_z / np.sum(exp_z, axis=1, keepdims=True)


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
        limit = sqrt(6 / n_in)
        weights.append(rng.uniform(-limit, limit, (n_in, n_out)))
        biases.append(np.zeros(n_out))
    return {"W": weights, "b": biases}

def forward(X, multilayer):
    activations = [X]
    A = X
    n_layers = len(multilayer["W"])
    for i, (W, b) in enumerate(zip(multilayer["W"], multilayer["b"])):
        Z = A @ W + b
        A = softmax(Z) if i == n_layers - 1 else sigmoid(Z)
        activations.append(A)
    return activations

def cross_entropy(P, Y):
    epsilon = 1e-15
    P = np.clip(P, epsilon, 1. - epsilon)
    
    return -np.mean(np.sum(Y * np.log(P), axis=1))

def binary_cross_entropy(P, Y):
    epsilon = 1e-15
    P = np.clip(P, epsilon, 1. - epsilon)
    
    return -np.mean(Y * np.log(P) + (1 - Y) * np.log(1 - P))

def backward(activations, Y, multilayer, n_layers):
    dW = [None] * n_layers
    db = [None] * n_layers
    m = Y.shape[0]
    delta = activations[-1] - Y
    for l, (W ,b) in reversed(list(enumerate(zip(multilayer["W"], multilayer["b"])))):
        dW[l] = (activations[l].T @ delta) / m
        db[l] = delta.sum(axis=0) / m
        if l > 0:
            A = activations[l]
            delta = (delta @ W.T) * A * (1 - A)
    return dW, db

def evaluate(X, y, multilayer):
    P = forward(X, multilayer)[-1]
    loss = cross_entropy(P, y)
    acc = np.mean(np.argmax(P, axis=1) == np.argmax(y, axis=1))
    return loss, acc

def mini_batch(config, multilayer, X, y, X_test, y_test):
    alpha = config.get("learning_rate")
    num_epochs = config.get("epochs")
    batch_size = config.get("batch_size")
    m = len(y)
    history = {"loss": [], "acc": [], "val_loss": [], "val_acc": []}
    for epoch in range(num_epochs):
        indices = np.random.permutation(m)
        X_shuffled = X[indices]
        y_shuffled = y[indices]
        for start_idx in range(0, m, batch_size):
            end_idx = min(start_idx + batch_size, m)
            X_batch = X_shuffled[start_idx:end_idx]
            y_batch = y_shuffled[start_idx:end_idx]
            n_layers = len(multilayer["W"])
            acts = forward(X_batch, multilayer)
            dW, db = backward(acts, y_batch, multilayer, n_layers)
            for l, (W, b) in enumerate(zip(multilayer["W"], multilayer["b"])):
                W -= alpha * dW[l] ; b -= alpha * db[l]
        loss, acc = evaluate(X, y, multilayer)
        val_loss, val_acc = evaluate(X_test, y_test, multilayer)
        for key, value in zip(history, (loss, acc, val_loss, val_acc)):
            history[key].append(value)
        print(f"epoch {epoch + 1:02d}/{num_epochs} - loss: {loss:.4f} - val_loss: {val_loss:.4f}"
              f" - acc: {acc:.4f} - val_acc: {val_acc:.4f}")
    return multilayer, history


def train_models(df, df_test,config):
    # clean_datasets = cleaning_datasets(datasets, config)

    norm_df, norm_stats, feature_cols = fit_normalizer(df)
    norm_df_test, norm_stats_tets, feature_cols_test = fit_normalizer(df_test)

    print(f"Training models with config: {config}")

    diag = (norm_df["diagnosis"] == "M").astype(np.float64).to_numpy()
    y = np.column_stack((diag ,1 - diag))
    X = norm_df[feature_cols].to_numpy(dtype=np.float64)
    diag_test = (norm_df_test["diagnosis"] == "M").astype(np.float64).to_numpy()
    y_test = np.column_stack((diag_test ,1 - diag_test))
    X_test = norm_df_test[feature_cols_test].to_numpy(dtype=np.float64)
    input_dim = X.shape[1]
    hidden_dims = config.get("layer")
    output_dim = 2
    sizes = [input_dim] + hidden_dims + [output_dim]
    multilayer = init_network(sizes)
    multilayer, history = mini_batch(config, multilayer, X, y, X_test, y_test)