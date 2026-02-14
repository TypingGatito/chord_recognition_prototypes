import argparse
import numpy as np
import sounddevice as sd
import soundfile as sf
import tempfile
import time
import os
from distance.recognition import chord_recognition

SR = 44100
BUFFER_SECONDS = 2.0
HOP_SECONDS = 0.5

BUFFER_SIZE = int(SR * BUFFER_SECONDS)
HOP_SIZE = int(SR * HOP_SECONDS)

audio_buffer = np.zeros(BUFFER_SIZE, dtype=np.float32)
write_ptr = 0

def audio_callback(indata, frames, time_info, status):
    global audio_buffer
    if status:
        print(status)

    samples = indata[:, 0]
    n = len(samples)
    audio_buffer[:] = np.roll(audio_buffer, -n)
    audio_buffer[-n:] = samples

def process_audio_buffer():
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        wav_path = tmp.name
    sf.write(wav_path, audio_buffer, SR)

    try:
        chords = chord_recognition(
            wav_path,
            num_harmonics=1,
            measure="EUC",
            filtering="median",
            L=17
        )
        if chords:
            print(f"\r🎶 {chords[-1]:10s}", end="")
    finally:
        os.remove(wav_path)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", "-f", help="Path to audio WAV file")
    args = parser.parse_args()

    if args.file:
        print(f"📂 Processing file: {args.file}")
        chords = chord_recognition(
            args.file,
            num_harmonics=1,
            measure="KL2",
            filtering="LP+M",
            L=17
        )
        for c in chords:
            print(c)
        return

    print("🎤 Recording from microphone (Ctrl+C to exit)")
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
            print("\n🛑 Stopped.")

if __name__ == "__main__":
    main()
