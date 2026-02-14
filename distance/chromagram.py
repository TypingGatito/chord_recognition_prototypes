import numpy as np
import librosa

def compute_chromagram(
    audio_path,
    alg="CQT1"
):
    if alg == "CQT":
        return compute_chromagram_cqt(
            audio_path =        audio_path,

        )
    else:
        return compute_chromagram_stft(
            audio_path=audio_path,

        )

def compute_chromagram_stft(
    audio_path,
    sr=44100,
    frame_length_ms=753,
    hop_length_ms=93
):
    y, sr = librosa.load(audio_path, sr=sr, mono=True)

    n_fft = int(sr * frame_length_ms / 1000)
    hop_length = int(sr * hop_length_ms / 1000)

    S = np.abs(
        librosa.stft(
            y,
            n_fft=n_fft,
            hop_length=hop_length,
            window="hann"
        )
    ) ** 2

    C = librosa.feature.chroma_stft(S=S, sr=sr)

    # Normalize each frame (important!)
    C /= np.maximum(C.sum(axis=0, keepdims=True), 1e-12)

    return C  # shape (12, N)


def compute_chromagram_cqt(
    audio_path,
    sr=44100,
    hop_length_ms=93,
    bins_per_octave=12,
    n_octaves=7
):
    # 1. Load audio
    y, sr = librosa.load(audio_path, sr=sr, mono=True)

    hop_length = int(sr * hop_length_ms / 1000)

    # 2. Constant-Q Transform
    CQT = np.abs(
        librosa.cqt(
            y,
            sr=sr,
            hop_length=hop_length,
            bins_per_octave=bins_per_octave,
            n_bins=bins_per_octave * n_octaves
        )
    ) ** 2  # power spectrum

    # 3. Pitch folding → chroma
    C = librosa.feature.chroma_cqt(
        C=CQT,
        sr=sr,
        hop_length=hop_length,
        bins_per_octave=bins_per_octave
    )

    # 4. Normalize each frame
    C /= np.maximum(C.sum(axis=0, keepdims=True), 1e-12)

    return C  # shape (12, N)
