import numpy as np
from .mathematical import apply_normalizer, forward, evaluate, binary_cross_entropy


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
    P = forward(X, multilayer)[-1]
    pred = np.argmax(P, axis=1)
    labels = np.where(pred == 0, "M", "B").tolist()
    loss, acc = evaluate(X, y, multilayer)
    bce = binary_cross_entropy(P[:, 0], y[:, 0])
    true_labels = df["diagnosis"].to_numpy()
    labels = np.where(pred == 0, "M", "B")
    wrong = np.flatnonzero(labels != true_labels)
    true_idx = np.where(true_labels == "M", 0, 1)
    for i in wrong:
        print(f"line {i}: predicted {labels[i]} ({P[i, pred[i]]:.1%}), "
      f"actual {true_labels[i]} (P({true_labels[i]}) = {P[i, true_idx[i]]:.1%})")
    print(f"total: {len(wrong)} errors on {len(true_labels)}")
    
    print(f"loss: {loss:.4f} - bce: {bce:.4f} - acc: {acc:.4f}")
