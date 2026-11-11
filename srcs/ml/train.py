import numpy as np
from math import sqrt
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
    A = X
    n_layers = len(multilayer["W"])
    for i, (W, b) in enumerate(zip(multilayer["W"], multilayer["b"])):
        Z = A @ W + b
        if i == n_layers - 1:
            A = softmax(Z)
        else:
            A = sigmoid(Z)
    return A

def mini_batch(config, multilayer, X, y):
    alpha = config.get("learning_rate")
    num_epochs = config.get("epochs")
    batch_size = config.get("batch_size")
    m = len(y)
    for epoch in range(num_epochs):
        indices = np.random.permutation(m)
        X_shuffled = X[indices]
        y_shuffled = y[indices]
        for start_idx in range(0, m, batch_size):
            end_idx = min(start_idx + batch_size, m)
            X_batch = X_shuffled[start_idx:end_idx]
            y_batch = y_shuffled[start_idx:end_idx]
            A = X_batch
            n_layers = len(multilayer["W"])
            for i, (W, b) in enumerate(zip(multilayer["W"], multilayer["b"])):
                Z = A @ W + b
                if i == n_layers - 1:
                    A = softmax(Z)
                else:
                    A = sigmoid(Z)





def train_models(df, config):
    # clean_datasets = cleaning_datasets(datasets, config)

    norm_df, norm_stats, feature_cols = fit_normalizer(df)

    print(f"Training models with config: {config}")

    diag = (norm_df["diagnosis"] == "M").astype(np.float64).to_numpy()
    y = np.column_stack((diag ,np.zeros_like(diag)))

    X = norm_df[feature_cols].to_numpy(dtype=np.float64)
    # print(y)
    input_dim = X.shape[1]
    hidden_dims = config.get("layer")
    output_dim = 2
    sizes = [input_dim] + hidden_dims + [output_dim]
    multilayer = init_network(sizes)
    for W, b in zip(multilayer["W"], multilayer["b"]):
        print(W.shape, b.shape)
    mini_batch(config, multilayer, X, y)