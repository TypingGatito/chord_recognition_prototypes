import numpy as np
import sounddevice as sd
import soundfile as sf
import time
import os
import librosa

SR = 44100
BUFFER_SECONDS = 2.0
HOP_SECONDS = 0.5

BUFFER_SIZE = int(SR * BUFFER_SECONDS)
HOP_SIZE = int(SR * HOP_SECONDS)


# ================= FILE MODE =================

def recognize_file(file_path: str, recognition_fn, **kwargs):
    y, sr = librosa.load(file_path, sr=SR, mono=True)
    return recognition_fn(y, sr, True, **kwargs)


# ================= TEST MODE =================

def recognize_test_first_20s(guitar_dir: str, recognition_fn, **kwargs):
    wav_files = [f for f in os.listdir(guitar_dir) if f.endswith(".wav")]
    if not wav_files:
        raise RuntimeError("No WAV files found in dataset.")

    wav_path = os.path.join(guitar_dir, wav_files[0])
    print(f"🧪 Testing on: {wav_path}")

    y, sr = sf.read(wav_path)
    y = y[:int(20 * sr)]

    return recognition_fn(y, sr, False, **kwargs)


# ================= MIC MODE =================

def recognize_microphone(recognition_fn, **kwargs):
    audio_buffer = np.zeros(BUFFER_SIZE, dtype=np.float32)

    def audio_callback(indata, frames, time_info, status):
        nonlocal audio_buffer

        if status:
            print(status)

        samples = indata[:, 0]
        n = len(samples)

        audio_buffer[:] = np.roll(audio_buffer, -n)
        audio_buffer[-n:] = samples

    def process_audio_buffer():
        chords = recognition_fn(audio_buffer.copy(), SR, True, **kwargs)

        if chords:
            last = chords[-1]
            if isinstance(last, tuple):
                label = last[2]
            else:
                label = last

            print(f"\r🎶 {label:10s}", end="")

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