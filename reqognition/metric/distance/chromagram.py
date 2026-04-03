import numpy as np
import librosa

def compute_chromagram_rt(y, sr=44100):
    frame_length = int(0.753 * sr)
    hop_length = int(0.093 * sr)

    S = np.abs(
        librosa.stft(
            y,
            n_fft=frame_length,
            hop_length=hop_length,
            window="hann"
        )
    ) ** 2

    C = librosa.feature.chroma_stft(S=S, sr=sr)
    C = C / np.max(C, axis=0, keepdims=True)

    return C, hop_length

def compute_chromagram_rt_cqt(
    y,
    sr=22050,
    hop_length=512
):
    C = librosa.feature.chroma_cqt(
        y=y,
        sr=sr,
        hop_length=hop_length,
        norm=None,
    )

    return C, hop_length