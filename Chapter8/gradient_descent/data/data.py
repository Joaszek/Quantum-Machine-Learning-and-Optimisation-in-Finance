import numpy as np

def xor_dataset():
    X = np.array([[0,0], [0,1], [1,0], [1,1]], dtype=float)
    y = np.array([-1, +1, +1, -1], dtype=float)
    return X, y