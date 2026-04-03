import numpy as np
import librosa
from typing import List, Tuple

# ---------------- CHORD SET ----------------
CHORDS = [f"{n}:maj" for n in ["C","C#","D","D#","E","F","F#","G","G#","A","A#","B"]] + \
         [f"{n}:min" for n in ["C","C#","D","D#","E","F","F#","G","G#","A","A#","B"]]

CHORD_TO_IDX = {c:i for i,c in enumerate(CHORDS)}

NOTE_IDX = {"C":0,"C#":1,"D":2,"D#":3,"E":4,"F":5,
            "F#":6,"G":7,"G#":8,"A":9,"A#":10,"B":11}

MODEL = None


# ---------------- IO ----------------
def load_lab_file(path):
    segs = []
    with open(path) as f:
        for line in f:
            s, e, lab = line.strip().split()
            segs.append((float(s), float(e), lab))
    return segs


# ---------------- CHROMA (TACTUS) ----------------
def extract_chroma_tactus(y, sr, hop=512):
    chroma = librosa.feature.chroma_cqt(y=y, sr=sr, hop_length=hop)
    chroma = librosa.util.normalize(chroma, axis=0)

    tempo, beats = librosa.beat.beat_track(y=y, sr=sr, hop_length=hop)
    beats = librosa.util.fix_frames(beats, x_min=0, x_max=chroma.shape[1])

    tactus = []
    times = []

    for i in range(len(beats)-1):
        s, e = beats[i], beats[i+1]
        if e > s:
            tactus.append(chroma[:, s:e].mean(axis=1))
            times.append(librosa.frames_to_time(s, sr=sr, hop_length=hop))

    tactus = np.array(tactus)
    tactus /= (np.linalg.norm(tactus, axis=1, keepdims=True) + 1e-6)

    return tactus, np.array(times)


# ---------------- SEGMENTS ----------------
def segments_to_frames(segments, times):
    labels = np.full(len(times), -1)
    for s, e, lab in segments:
        if lab not in CHORD_TO_IDX:
            continue
        idx = CHORD_TO_IDX[lab]
        mask = (times >= s) & (times < e)
        labels[mask] = idx
    return labels


# ---------------- CIRCLE OF FIFTHS ----------------
CIRCLE = ["C","G","D","A","E","B","F#","C#","G#","D#","A#","F"]
note_to_pos = {n:i for i,n in enumerate(CIRCLE)}

def circle_dist(n1, n2):
    d = abs(note_to_pos[n1] - note_to_pos[n2])
    return min(d, 12 - d)


def build_transition_matrix():
    N = len(CHORDS)
    A = np.zeros((N, N))

    for i, ci in enumerate(CHORDS):
        ni = ci.split(":")[0]
        for j, cj in enumerate(CHORDS):
            nj = cj.split(":")[0]
            d = circle_dist(ni, nj)
            A[i, j] = 12 - d

    A += 1e-3
    A /= A.sum(axis=1, keepdims=True)
    return A


# ---------------- EMISSION (FIXED TEMPLATES) ----------------
def build_templates():
    N = len(CHORDS)
    D = 12

    means = np.zeros((N, D))
    covs = np.zeros((N, D, D))

    for i, chord in enumerate(CHORDS):
        root, typ = chord.split(":")
        r = NOTE_IDX[root]

        if typ == "maj":
            triad = [r, (r+4)%12, (r+7)%12]
        else:
            triad = [r, (r+3)%12, (r+7)%12]

        means[i, triad] = 1.0
        means[i] /= means[i].sum() + 1e-12

        cov = np.zeros((D, D))

        for d in range(D):
            cov[d, d] = 0.2

        t, m, d_ = triad

        cov[t, t] = 1.0
        cov[m, m] = 1.0
        cov[d_, d_] = 1.0

        cov[t, d_] = cov[d_, t] = 0.8
        cov[m, d_] = cov[d_, m] = 0.8
        cov[t, m] = cov[m, t] = 0.6

        covs[i] = cov + 1e-6 * np.eye(D)

    return means, covs


# ---------------- TRAIN ----------------
def train_hmm_manual(audio_files: List[str], n_iter=10):
    global MODEL

    X_all = []

    for path in audio_files:
        y, sr = librosa.load(path, sr=None, mono=True)
        X, _ = extract_chroma_tactus(y, sr)
        if len(X) > 0:
            X_all.append(X)

    if len(X_all) == 0:
        raise RuntimeError("No training data")

    X = np.vstack(X_all)  # (T, 12)

    T, D = X.shape
    N = len(CHORDS)

    pi = np.ones(N) / N

    A = build_transition_matrix()

    means, covs = build_templates()

    for _ in range(n_iter):
        log_B = np.zeros((T, N))
        for j in range(N):
            log_B[:, j] = log_gauss(X, means[j], covs[j])

        log_pi = np.log(pi + 1e-12)
        log_A = np.log(A + 1e-12)

        alpha = np.zeros((T, N))
        alpha[0] = log_pi + log_B[0]

        for t in range(1, T):
            for j in range(N):
                alpha[t, j] = np.logaddexp.reduce(alpha[t-1] + log_A[:, j]) + log_B[t, j]

        beta = np.zeros((T, N))
        for t in range(T-2, -1, -1):
            for i in range(N):
                beta[t, i] = np.logaddexp.reduce(
                    log_A[i] + log_B[t+1] + beta[t+1]
                )

        log_P = np.logaddexp.reduce(alpha[-1])
        gamma = np.exp(alpha + beta - log_P)

        xi = np.zeros((T - 1, N, N))
        for t in range(T - 1):
            tmp = (
                    alpha[t][:, None]
                    + log_A
                    + log_B[t + 1][None, :]
                    + beta[t + 1][None, :]
                    - log_P
            )
            xi[t] = np.exp(tmp)

        pi = gamma[0]
        pi /= pi.sum()

        A = xi.sum(axis=0)
        A /= A.sum(axis=1, keepdims=True)

    MODEL = (np.log(pi+1e-12), np.log(A+1e-12), means, covs)


# ---------------- GAUSSIAN ----------------
def log_gauss(x, mean, cov):
    inv = np.linalg.inv(cov)
    diff = x - mean
    return -0.5 * (
        np.einsum('ij,jk,ik->i', diff, inv, diff) +
        np.log(np.linalg.det(cov) + 1e-12)
    )

# ---------------- VITERBI ----------------
def viterbi(log_emission, log_trans, log_init):
    T, N = log_emission.shape
    dp = np.zeros((T, N))
    back = np.zeros((T, N), dtype=int)

    dp[0] = log_init + log_emission[0]

    for t in range(1, T):
        for j in range(N):
            vals = dp[t-1] + log_trans[:, j]
            back[t, j] = np.argmax(vals)
            dp[t, j] = np.max(vals) + log_emission[t, j]

    path = np.zeros(T, dtype=int)
    path[-1] = np.argmax(dp[-1])

    for t in range(T-2, -1, -1):
        path[t] = back[t+1, path[t+1]]

    return path


# ---------------- RECOGNITION ----------------
def recognize_hmm_manual(y, sr) -> List[Tuple[float, float, str]]:
    global MODEL
    if MODEL is None:
        raise RuntimeError("Model not trained")

    chroma, times = extract_chroma_tactus(y, sr)

    T = len(chroma)
    N = len(CHORDS)

    log_init, log_trans, means, covs = MODEL

    K = min(4, len(chroma))

    log_B_init = np.zeros((K, N))
    for j in range(N):
        log_B_init[:, j] = log_gauss(chroma[:K], means[j], covs[j])

    log_init = log_B_init.sum(axis=0)
    log_init -= np.logaddexp.reduce(log_init)

    # smoothing
    log_uniform = np.log(np.ones(N) / N)
    log_init = 0.8 * log_init + 0.2 * log_uniform


    log_emission = np.zeros((T, N))
    for c in range(N):
        log_emission[:, c] = log_gauss(chroma, means[c], covs[c])

    path = viterbi(log_emission, log_trans, log_init)

    res = []
    start = times[0]
    prev = path[0]

    for i in range(1, T):
        if path[i] != prev:
            res.append((start, times[i], CHORDS[prev]))
            start = times[i]
            prev = path[i]

    res.append((start, times[-1], CHORDS[prev]))
    return res