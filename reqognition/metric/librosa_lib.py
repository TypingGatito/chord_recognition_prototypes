import numpy as np
import librosa
from scipy.ndimage import median_filter
from typing import List, Tuple

# ==========================
# CONFIG
# ==========================

HOP_LENGTH = 512

# ==========================
# TEMPLATES (FMP)
# ==========================

pitch_classes = ['C','C#','D','D#','E','F','F#','G','G#','A','A#','B']

template_maj = np.array([1,0,0,0,1,0,0,1,0,0,0,0])
template_min = np.array([1,0,0,1,0,0,0,1,0,0,0,0])

templates = []
labels = []

for i, name in enumerate(pitch_classes):
    templates.append(np.roll(template_maj, i))
    labels.append(f"{name}:maj")

for i, name in enumerate(pitch_classes):
    templates.append(np.roll(template_min, i))
    labels.append(f"{name}:min")

templates = np.array(templates, dtype=float)

# L1 normalization
templates = templates / np.sum(templates, axis=1, keepdims=True)

# ==========================
# FMP METRIC RECOGNITION
# ==========================

def recognize_from_signal(y: np.ndarray, sr: int) -> List[Tuple[float, float, str]]:

    # harmonic component
    y = librosa.effects.harmonic(y)

    # CQT chroma
    chroma = librosa.feature.chroma_cqt(
        y=y,
        sr=sr,
        hop_length=HOP_LENGTH,
        norm=None
    ).T

    # L1 normalization
    chroma = chroma / (np.sum(chroma, axis=1, keepdims=True) + 1e-6)

    # L2 distance
    diff = chroma[:, None, :] - templates[None, :, :]
    dist = np.linalg.norm(diff, axis=2)
    path = np.argmin(dist, axis=1)

    # median smoothing
    path = median_filter(path, size=9)

    # segment creation
    hop_time = HOP_LENGTH / sr
    chords = []

    start = 0.0
    current = path[0]

    for i in range(1, len(path)):
        if path[i] != current:
            end = i * hop_time
            if end - start > 0.2:
                chords.append((start, end, labels[current]))
            start = end
            current = path[i]

    end = len(path) * hop_time
    if end - start > 0.2:
        chords.append((start, end, labels[current]))

    return chords


# ==========================
# PUBLIC INTERFACE
# ==========================

def recognize_fmp(y, sr):
    """
    Runs FMP metric template chord recognition on audio signal.

    Returns:
        List[(start, end, chord)]
    """
    return recognize_from_signal(y, sr)