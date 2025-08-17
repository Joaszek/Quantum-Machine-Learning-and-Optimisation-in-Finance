import numpy as np

def mse_loss(preds, targets):
    preds = np.asarray(preds,dtype=float)
    targets = np.asarray(targets, dtype=float)
    return float(np.mean((preds-targets) ** 2))