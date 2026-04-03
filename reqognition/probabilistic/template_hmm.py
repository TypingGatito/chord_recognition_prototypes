import numpy as np
import librosa
from typing import List, Tuple


# ==========================
# CONFIG
# ==========================

HOP = 1024

PITCH = ['C','C#','D','D#','E','F','F#','G','G#','A','A#','B']

CHORDS = [f"{n}:maj" for n in PITCH] + [f"{n}:min" for n in PITCH]
N = len(CHORDS)


# ==========================
# TEMPLATES
# ==========================

maj = np.array([1,0,0,0,1,0,0,1,0,0,0,0])
min_ = np.array([1,0,0,1,0,0,0,1,0,0,0,0])

TEMPLATES = []
for i in range(12):
    TEMPLATES.append(np.roll(maj, i))
for i in range(12):
    TEMPLATES.append(np.roll(min_, i))

TEMPLATES = np.array(TEMPLATES, float)
TEMPLATES /= np.linalg.norm(TEMPLATES, axis=1, keepdims=True)


# ==========================
# FEATURE EXTRACTION (NO TACTUS)
# ==========================

def extract_chroma(y, sr):
    chroma = librosa.feature.chroma_cqt(
        y=y,
        sr=sr,
        hop_length=HOP
    )
    chroma = librosa.util.normalize(chroma, axis=0)
    return chroma.T  # (T, 12)


# ==========================
# TRANSITION (SIMPLE, STABLE)
# ==========================

def build_A(p):
    A = np.ones((N, N)) * ((1 - p) / (N - 1))
    np.fill_diagonal(A, p)
    return A


# ==========================
# VITERBI
# ==========================

def viterbi(logB, logA, logC):
    T, N = logB.shape
    dp = np.zeros((T, N))
    back = np.zeros((T, N), dtype=int)

    dp[0] = logC + logB[0]

    for t in range(1, T):
        for j in range(N):
            vals = dp[t-1] + logA[:, j]
            back[t, j] = np.argmax(vals)
            dp[t, j] = np.max(vals) + logB[t, j]

    path = np.zeros(T, dtype=int)
    path[-1] = np.argmax(dp[-1])

    for t in range(T-2, -1, -1):
        path[t] = back[t+1, path[t+1]]

    return path


# ==========================
# MAIN
# ==========================

def recognize(y, sr) -> List[Tuple[float, float, str]]:
    X = extract_chroma(y, sr)   # (T, 12)

    B = TEMPLATES @ X.T
    B = np.exp( B)
    B = B.T + 1e-12

    logB = np.log(B)

    A = build_A(p=0.2)
    logA = np.log(A)

    C = np.ones(N) / N
    logC = np.log(C)

    path = viterbi(logB, logA, logC)

    hop_time = HOP / sr

    res = []
    start = 0
    prev = path[0]

    for i in range(1, len(path)):
        if path[i] != prev:
            res.append((start * hop_time, i * hop_time, CHORDS[prev]))
            start = i
            prev = path[i]

    res.append((start * hop_time, len(path) * hop_time, CHORDS[prev]))

    return res


def recognize_hmm(y,
                  sr,
                  isOnline=False,
                  ) -> List[Tuple[float, float, str]]:

    return recognize(y, sr)