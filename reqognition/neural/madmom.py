import tempfile
import soundfile as sf
import os

from madmom.features.chords import (
    CNNChordFeatureProcessor,
    CRFChordRecognitionProcessor
)
# ================= INITIALIZE PROCESSORS =================

cnn_processor = CNNChordFeatureProcessor()
crf_processor = CRFChordRecognitionProcessor()


# ================= PUBLIC INTERFACE =================

def recognize_madmom(y, sr, is_online):
    """
    Runs Madmom CNN+CRF chord recognition on audio signal.

    Returns:
        List[(start, end, chord)]
    """

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        path = tmp.name

    sf.write(path, y, sr)

    try:
        features = cnn_processor(path)
        chords = crf_processor(features)
    finally:
        os.remove(path)

    if chords is None or len(chords) == 0:
        return []

    return [(float(start), float(end), label) for start, end, label in chords]

def recognize_madmom_file(path: str):
    features = cnn_processor(path)
    chords = crf_processor(features)

    if chords is None or len(chords) == 0:
        return []

    return [(float(start), float(end), label) for start, end, label in chords]