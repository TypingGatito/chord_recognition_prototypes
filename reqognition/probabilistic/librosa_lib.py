from typing import List, Tuple

import librosa
import numpy as np

# ==========================
# CONFIG
# ==========================

HOP_LENGTH = 1024

# ==========================
# CHORD TEMPLATES (24)
# ==========================

PITCH_CLASS_NAMES = ['C','C#','D','D#','E','F','F#','G','G#','A','A#','B']

CHORD_LABELS = []
TEMPLATES = []

maj = np.array([1,0,0,0,1,0,0,1,0,0,0,0])
min = np.array([1,0,0,1,0,0,0,1,0,0,0,0])

for i, name in enumerate(PITCH_CLASS_NAMES):
    TEMPLATES.append(np.roll(maj, i))
    CHORD_LABELS.append(f"{name}:maj")

for i, name in enumerate(PITCH_CLASS_NAMES):
    TEMPLATES.append(np.roll(min, i))
    CHORD_LABELS.append(f"{name}:min")

TEMPLATES = np.array(TEMPLATES).astype(float)

# L2 normalization
TEMPLATES = TEMPLATES / np.linalg.norm(TEMPLATES, axis=1, keepdims=True)

N_CHORDS = len(CHORD_LABELS)

# ==========================
# FEATURE EXTRACTION
# ==========================

def extract_chroma(y, sr):

    chroma = librosa.feature.chroma_cqt(
        y=y,
        sr=sr,
        hop_length=HOP_LENGTH
    )

    chroma = chroma / (np.linalg.norm(chroma, axis=0, keepdims=True) + 1e-6)

    return chroma


# ==========================
# TEMPLATE SIMILARITY
# ==========================

def compute_similarity(chroma):

    return TEMPLATES @ chroma


# ==========================
# TRANSITION MATRIX
# ==========================

def uniform_transition_matrix(p):

    I = N_CHORDS

    A = np.ones((I, I)) * ((1 - p) / (I - 1))
    np.fill_diagonal(A, p)

    return A


# ==========================
# VITERBI (log likelihood)
# ==========================

def viterbi_log(A, C, B):

    N = B.shape[1]
    I = B.shape[0]

    D = np.zeros((I, N))
    E = np.zeros((I, N), dtype=int)

    D[:,0] = np.log(C) + np.log(B[:,0] + 1e-12)

    for n in range(1, N):

        for j in range(I):

            prob = D[:,n-1] + np.log(A[:,j] + 1e-12)

            E[j,n] = np.argmax(prob)

            D[j,n] = np.max(prob) + np.log(B[j,n] + 1e-12)

    path = np.zeros(N, dtype=int)

    path[-1] = np.argmax(D[:,-1])

    for n in range(N-2, -1, -1):
        path[n] = E[path[n+1], n+1]

    return path


# ==========================
# MAIN RECOGNITION
# ==========================

def recognize_from_signal(y, sr):

    chroma = extract_chroma(y, sr)

    chord_sim = compute_similarity(chroma)

    # emission
    B = chord_sim + 1e-6

    # HMM parameters
    A = uniform_transition_matrix(p=0.15)
    C = np.ones((N_CHORDS)) / N_CHORDS

    path = viterbi_log(A, C, B)

    hop_time = HOP_LENGTH / sr

    segments = []

    start = 0
    prev = path[0]

    for i in range(1, len(path)):

        if path[i] != prev:

            end = i * hop_time
            segments.append((start * hop_time, end, CHORD_LABELS[prev]))

            start = i
            prev = path[i]

    segments.append((start * hop_time, len(path)*hop_time, CHORD_LABELS[prev]))

    return segments


# ==========================
# PUBLIC API
# ==========================

def recognize_hmm(y,
                  sr,
                  isOnline=False,
                  ) -> List[Tuple[float, float, str]]:

    return recognize_from_signal(y, sr)