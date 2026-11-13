import numpy as np
from math import sqrt, log
from pandas import DataFrame
from matplotlib import pyplot as plt
import json


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
    print(n_layers)

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