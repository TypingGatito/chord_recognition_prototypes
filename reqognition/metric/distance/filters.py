from scipy.signal import medfilt
import numpy as np

def low_pass_filter(D, L):
    kernel = np.ones(L) / L
    return np.apply_along_axis(
        lambda x: np.convolve(x, kernel, mode="same"),
        axis=1,
        arr=D
    )


def low_pass_filter_online(D, L):
    """
    Causal (online) low-pass filter.
    D: shape (K, N) – distance matrix
    L: window length
    """
    K, N = D.shape
    D_out = np.zeros_like(D)

    for k in range(K):
        for n in range(N):
            start = max(0, n - L + 1)
            window = D[k, start:n + 1]
            D_out[k, n] = np.mean(window)

    return D_out

def median_filter_offline(D, L):
    return np.apply_along_axis(
        lambda x: medfilt(x, kernel_size=L),
        axis=1,
        arr=D
    )

def median_filter(D, L):
    """
    Causal (realtime) median filter.
    D: shape (K, N) — distance matrix
    L: window length
    """
    K, N = D.shape
    D_out = np.zeros_like(D)

    for k in range(K):
        for n in range(N):
            start = max(0, n - L + 1)
            window = D[k, start:n + 1]
            D_out[k, n] = np.median(window)

    return D_out