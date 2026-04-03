import numpy as np

def optimal_scale(c, p):
    denom = np.dot(c, c)
    if denom < 1e-12:
        return 0.0
    return np.dot(c, p) / denom

def optimal_scale_kl(c, p):
    denom = np.sum(c)
    if denom < 1e-12:
        return 0.0
    return np.sum(p) / denom