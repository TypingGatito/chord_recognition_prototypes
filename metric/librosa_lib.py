import argparse
import numpy as np
import sounddevice as sd
import soundfile as sf
import tempfile
import time
import os
import threading
import librosa

# ==========================
# CONFIG
# ==========================

SR = 22050
BUFFER_SECONDS = 2.0
HOP_SECONDS = 0.5

BUFFER_SIZE = int(SR * BUFFER_SECONDS)
HOP_SIZE = int(SR * HOP_SECONDS)

audio_buffer = np.zeros(BUFFER_SIZE, dtype=np.float32)
buffer_lock = threading.Lock()

# ==========================
# CHORD TEMPLATES
# ==========================

PITCH_CLASS_NAMES = ['C','C#','D','D#','E','F','F#','G','G#','A','A#','B']

CHORD_LABELS = []
TEMPLATES = []

maj_template = np.array([1,0,0,0,1,0,0,1,0,0,0,0])
min_template = np.array([1,0,0,1,0,0,0,1,0,0,0,0])

# Major
for i, name in enumerate(PITCH_CLASS_NAMES):
    TEMPLATES.append(np.roll(maj_template, i))
    CHORD_LABELS.append(f"{name}:maj")

# Minor
for i, name in enumerate(PITCH_CLASS_NAMES):
    TEMPLATES.append(np.roll(min_template, i))
    CHORD_LABELS.append(f"{name}:min")

# No chord
TEMPLATES.append(np.zeros(12))
CHORD_LABELS.append("N")

TEMPLATES = np.array(TEMPLATES)

# normalize templates
TEMPLATES = TEMPLATES / (np.linalg.norm(TEMPLATES, axis=1, keepdims=True)+1e-6)

# ==========================
# AUDIO CALLBACK
# ==========================

def audio_callback(indata, frames, time_info, status):

    if status:
        print(status)

    samples = indata[:,0]

    with buffer_lock:

        n = len(samples)

        audio_buffer[:-n] = audio_buffer[n:]
        audio_buffer[-n:] = samples

# ==========================
# FEATURE EXTRACTION
# ==========================

def extract_chroma(y):

    chroma = librosa.feature.chroma_stft(
        y=y,
        sr=SR,
        n_fft=2048,
        hop_length=512
    )

    chroma = chroma.T

    chroma = chroma / (np.linalg.norm(chroma, axis=1, keepdims=True)+1e-6)

    return chroma

# ==========================
# TEMPLATE MATCHING
# ==========================

def template_recognition(chroma):

    similarity = chroma @ TEMPLATES.T

    chord_indices = np.argmax(similarity, axis=1)

    return chord_indices

# ==========================
# CHORD RECOGNITION
# ==========================

def recognize_chords(wav_path):

    y, sr = librosa.load(wav_path, sr=SR)

    chroma = extract_chroma(y)

    path = template_recognition(chroma)

    chords = []

    hop = 512 / SR

    start = 0
    current = path[0]

    for i in range(1, len(path)):

        if path[i] != current:

            end = i * hop

            chords.append((start, end, CHORD_LABELS[current]))

            start = end
            current = path[i]

    chords.append((start, len(path)*hop, CHORD_LABELS[current]))

    return chords

# ==========================
# REALTIME PROCESSING
# ==========================

def process_audio_buffer():

    with buffer_lock:
        buffer_copy = audio_buffer.copy()

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        wav_path = tmp.name

    sf.write(wav_path, buffer_copy, SR)

    try:

        chords = recognize_chords(wav_path)

        if len(chords) > 0:

            last = chords[-1][2]

            print(f"\r🎶 {last:10s}", end="", flush=True)

    finally:

        os.remove(wav_path)

# ==========================
# FILE MODE
# ==========================

def process_file(path):

    chords = recognize_chords(path)

    for start, end, chord in chords:

        print(f"{start:.2f} - {end:.2f}: {chord}")

# ==========================
# MAIN
# ==========================

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--file", "-f")

    args = parser.parse_args()

    if args.file:

        print(f"📂 Processing file: {args.file}")
        process_file(args.file)
        return

    print("🎤 Template chord recognition realtime")

    with sd.InputStream(
        channels=1,
        samplerate=SR,
        blocksize=HOP_SIZE,
        callback=audio_callback
    ):

        try:

            while True:

                time.sleep(HOP_SECONDS)

                process_audio_buffer()

        except KeyboardInterrupt:

            print("\nStopped")

if __name__ == "__main__":
    main()
