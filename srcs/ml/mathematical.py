import numpy as np
from math import sqrt, log
from pandas import DataFrame
from matplotlib import pyplot as plt
import json
import sys
from pathlib import Path

CURRENT_DIR = Path(__file__).resolve().parent
SRCS_DIR = CURRENT_DIR.parent
if str(SRCS_DIR) not in sys.path:
    sys.path.insert(0, str(SRCS_DIR))
from utils import TrainState

def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def softmax(z):
    exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))
    return exp_z / np.sum(exp_z, axis=1, keepdims=True)

def apply_normalizer(df, stats, feature_cols):
    """Applique la normalisation du train sur le jeu de test."""
    norm_df = df.copy()
    for col in feature_cols:

        values = norm_df[col].astype(np.float64)
        mean = stats[col]["mean"]
        std = stats[col]["std"]
        norm_df[col] = (values - mean) / (std + 1e-8)
    return norm_df

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

def metrics(state: TrainState, pred):
    P_valid = forward(state.X_valid, state.multilayer)[-1]

    # pred_m = np.argmax(P_valid, axis=1) == 0
    # true_m = np.argmax(state.y_valid, axis=1) == 0
    # pred_n = np.argmax(P_valid, axis=1) == 1
    # true_n = np.argmax(state.y_valid, axis=1) == 1
    true_indices = np.argmax(state.y_valid, axis=1)
    pred_indices = pred if pred.ndim == 1 else np.argmax(pred, axis=1)

    labels = np.where(pred_indices == 0, "M", "B")
    true_labels = np.where(true_indices == 0, "M", "B")

    TP = FP = TN = FN = 0

    for i in range(len(true_labels)):
        if labels[i] == "M" and true_labels[i] == "M":
            TP += 1
        elif labels[i] == "B" and true_labels[i] == "B":
            TN += 1
        elif labels[i] == "M" and true_labels[i] == "B":
            FP += 1
        elif labels[i] == "B" and true_labels[i] == "M":
            FN += 1

    precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
    rappel = TP / (TP + FN) if (TP + FN) > 0 else 0.0
    F1_score = 2 * precision * rappel / (precision + rappel) if precision + rappel > 0 else 0.0

    return precision, rappel, F1_score




def evaluate(state: TrainState):
    P_train = forward(state.X_train, state.multilayer)[-1]
    loss = cross_entropy(P_train, state.y_train)
    acc = np.mean(np.argmax(P_train, axis=1) == np.argmax(state.y_train, axis=1))
    
    P_valid = forward(state.X_valid, state.multilayer)[-1]
    val_loss = cross_entropy(P_valid, state.y_valid)
    val_acc = np.mean(np.argmax(P_valid, axis=1) == np.argmax(state.y_valid, axis=1))

    precision, rappel, F1_score = metrics(state, P_valid)
    return {"loss": loss, "acc": acc, "val_loss": val_loss, "val_acc": val_acc, "precision": precision, "rappel": rappel, "F1_score": F1_score}