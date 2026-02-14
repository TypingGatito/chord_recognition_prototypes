import argparse
import numpy as np
import sounddevice as sd
import soundfile as sf
import tempfile
import time
import os
import threading


from madmom.features.chords import (
    CNNChordFeatureProcessor,
    CRFChordRecognitionProcessor
)

SR = 44100
BUFFER_SECONDS = 2.0
HOP_SECONDS = 0.5

BUFFER_SIZE = int(SR * BUFFER_SECONDS)
HOP_SIZE = int(SR * HOP_SECONDS)

audio_buffer = np.zeros(BUFFER_SIZE, dtype=np.float32)
buffer_lock = threading.Lock()

# initialize madmom processors ONCE (important for speed)
cnn_processor = CNNChordFeatureProcessor()
crf_processor = CRFChordRecognitionProcessor()


def audio_callback(indata, frames, time_info, status):
    global audio_buffer

    if status:
        print(status)

    samples = indata[:, 0]
    n = len(samples)

    with buffer_lock:
        audio_buffer[:] = np.roll(audio_buffer, -n)
        audio_buffer[-n:] = samples


def recognize_chords(wav_path):
    features = cnn_processor(wav_path)
    chords = crf_processor(features)
    return chords


def process_audio_buffer():
    # copy buffer safely
    with buffer_lock:
        buffer_copy = audio_buffer.copy()

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        wav_path = tmp.name

    sf.write(wav_path, buffer_copy, SR)

    try:
        chords = recognize_chords(wav_path)

        # FIX: proper numpy array check
        if chords is not None and len(chords) > 0:
            last_chord = chords[-1][2]
            print(f"\r🎶 {last_chord:10s}", end="", flush=True)

    except Exception as e:
        print(f"\nError: {e}")

    finally:
        try:
            os.remove(wav_path)
        except:
            pass


def process_file(path):
    chords = recognize_chords(path)

    print("DEBUG chords:", chords)
    print("DEBUG len:", 0 if chords is None else len(chords))

    if chords is not None and len(chords) > 0:
        for start, end, label in chords:
            print(f"{start:.2f} - {end:.2f}: {label}")
    else:
        print("No chords detected")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", "-f", help="Path to audio file")
    args = parser.parse_args()

    if args.file:
        print(f"📂 Processing file: {args.file}")
        process_file(args.file)
        return

    print("🎤 Recording from microphone (Ctrl+C to exit)")

    with sd.InputStream(
        channels=1,
        samplerate=SR,
        blocksize=HOP_SIZE,
        callback=audio_callback,
        dtype=np.float32
    ):
        try:
            while True:
                time.sleep(HOP_SECONDS)
                process_audio_buffer()

        except KeyboardInterrupt:
            print("\n🛑 Stopped.")


if __name__ == "__main__":
    main()
