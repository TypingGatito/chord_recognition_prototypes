import numpy as np

def euclidean_distance(x, y):
    return np.sum((x - y) ** 2)

def kl_divergence(x, y):
    eps = 1e-12
    return np.sum(x * np.log((x + eps) / (y + eps)) - x + y)