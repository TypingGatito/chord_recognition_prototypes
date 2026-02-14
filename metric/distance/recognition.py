from .scale import *
from .distances import *
from .chromagram import *
from .filters import *
from .base_generation import *

def recognize_chords_framewise(C, templates, measure="EUC"):
    chord_names = list(templates.keys())
    P = np.stack([templates[k] for k in chord_names])

    N = C.shape[1]
    K = len(chord_names)

    D = np.zeros((K, N))

    for n in range(N):
        c = C[:, n]

        for k in range(K):
            p = P[k]
            h = optimal_scale(c, p)
            hc = h * c

            if measure == "EUC":
                D[k, n] = euclidean_distance(hc, p)
            elif measure == "KL2":
                D[k, n] = kl_divergence(p, hc)

    return chord_names, D

def decode_chords(D):
    return np.argmin(D, axis=0)

def chord_recognition(
    audio_path,
    num_harmonics=1,
    measure="EUC",
    filtering="LP+M",
    L=17
):
    C = compute_chromagram(audio_path)
    templates = generate_all_templates(num_harmonics)

    chord_names, D = recognize_chords_framewise(C, templates, measure)

    if filtering == "lowpass":
        D = low_pass_filter(D, L)
    elif filtering == "median":
        D = median_filter(D, L)
    elif filtering == "LP+M":
        D = median_filter(low_pass_filter(D, 5), 17)

    idx = decode_chords(D)
    return [chord_names[i] for i in idx]
