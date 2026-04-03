from .scale import *
from .distances import *
from .chromagram import *
from .filters import *
from .base_generation import *

def recognize_chords_framewise(C, templates, measure="KL2"):
    chord_names = list(templates.keys())
    P = np.stack([templates[k] for k in chord_names])

    N = C.shape[1]
    K = len(chord_names)

    D = np.zeros((K, N))

    for n in range(N):
        c = C[:, n]

        for k in range(K):
            p = P[k]
            if measure == "EUC":
                h = optimal_scale(c, p)
            elif measure == "KL2":
                h = optimal_scale_kl(c, p)
            hc = h * c

            if measure == "EUC":
                D[k, n] = euclidean_distance(hc, p)
            elif measure == "KL2":
                D[k, n] = kl_divergence(p, hc)

    return chord_names, D

def decode_chords(D):
    return np.argmin(D, axis=0)

def chord_recognition_dist(
    y,
    sr,
    isOnline=False,
    num_harmonics=1,
    measure="KL2",
    filtering="LP+M",
    L=17
):
    C, hop_length = compute_chromagram_rt_cqt(y, sr)

    templates = generate_all_templates(num_harmonics)

    chord_names, D = recognize_chords_framewise(C, templates, measure)

    if filtering == "lowpass":
        D = low_pass_filter(D, L)
    elif filtering == "median":
        D = median_filter(D, L)
    elif filtering == "LP+M":
        if isOnline:
            D = median_filter(low_pass_filter_online(D, 5), 17)
        else:
            D = median_filter_offline(low_pass_filter(D, 5), 17)

    return D

def chord_recognition_rt(
    y,
    sr,
    isOnline=False,
    num_harmonics=1,
    measure="KL2",
    filtering="LP+M",
    L=17
):
    C, hop_length = compute_chromagram_rt_cqt(y, sr)

    templates = generate_all_templates(num_harmonics)

    chord_names, D = recognize_chords_framewise(C, templates, measure)

    if filtering == "lowpass":
        D = low_pass_filter(D, L)
    elif filtering == "median":
        D = median_filter(D, L)
    elif filtering == "LP+M":
        if isOnline:
            D = median_filter(low_pass_filter_online(D, 5), 17)
        else:
            D = median_filter_offline(low_pass_filter(D, 5), 17)

    idx = decode_chords(D)

    hop_time = hop_length / sr

    chords = [chord_names[i] for i in idx]

    return chords, hop_time
