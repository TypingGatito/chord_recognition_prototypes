import os
import tempfile
import soundfile as sf
from typing import List, Tuple

from omnizart.chord.app import ChordTranscription


NOTE_NAMES = [
    "C","C#","D","D#","E","F",
    "F#","G","G#","A","A#","B"
]


def detect_chord(pitches):

    pcs = sorted({p % 12 for p in pitches})

    for root in pcs:

        intervals = sorted((p - root) % 12 for p in pcs)

        if intervals == [0,4,7]:
            return NOTE_NAMES[root] + ":maj"

        if intervals == [0,3,7]:
            return NOTE_NAMES[root] + ":min"

    return NOTE_NAMES[pcs[0]]


chord_model = ChordTranscription()


# ================= AUDIO ARRAY VERSION =================

def recognize_omnizart(y, sr, is_online) -> List[Tuple[float, float, str]]:

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        path = tmp.name

    sf.write(path, y, sr)

    try:
        chords = recognize_omnizart_file(path)
    finally:
        os.remove(path)

    return chords


# ================= FILE VERSION =================

def recognize_omnizart_file(file_path: str) -> List[Tuple[float, float, str]]:

    midi = chord_model.transcribe(file_path, output=None)

    notes = []

    for inst in midi.instruments:
        for n in inst.notes:
            notes.append((n.start, n.end, n.pitch))

    if not notes:
        return []

    notes.sort(key=lambda x: x[0])

    chords = []
    i = 0

    while i < len(notes):

        start = notes[i][0]
        end = notes[i][1]

        pitches = []

        while i < len(notes) and abs(notes[i][0] - start) < 1e-3:
            pitches.append(notes[i][2])
            end = max(end, notes[i][1])
            i += 1

        chord = detect_chord(pitches)

        chords.append(
            (
                float(start),
                float(end),
                chord
            )
        )

    return chords