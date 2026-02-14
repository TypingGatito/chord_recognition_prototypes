import numpy as np
import libfmp.c4
import libfmp.c5
from distance.recognition import chord_recognition


def convert_labels_to_matrix(labels, chord_labels):

    K = len(chord_labels)
    N = len(labels)

    chord_matrix = np.zeros((K, N))

    label_to_index = {
        chord_labels[i]: i
        for i in range(K)
    }

    for n, label in enumerate(labels):

        root = label.split(":")[0]

        if "min" in label:
            label_norm = root + "m"
        else:
            label_norm = root

        if label_norm in label_to_index:
            chord_matrix[label_to_index[label_norm], n] = 1

    return chord_matrix


def evaluate_song_with_your_method(
        audio_path,
        annotation_path,
        chord_recognition_func
):

    print("\nProcessing:", audio_path)

    chord_labels = libfmp.c5.get_chord_labels(ext_minor='m')

    ann_matrix, _ = libfmp.c5.convert_chord_ann_matrix(
        annotation_path,
        chord_labels,
        Fs=1,
        N=None,
        last=False
    )

    predicted_labels = chord_recognition_func(audio_path)

    chord_matrix = convert_labels_to_matrix(
        predicted_labels,
        chord_labels
    )

    N = min(
        ann_matrix.shape[1],
        chord_matrix.shape[1]
    )

    ann_matrix = ann_matrix[:, :N]
    chord_matrix = chord_matrix[:, :N]

    result = libfmp.c5.compute_eval_measures(
        ann_matrix,
        chord_matrix
    )

    P, R, F, TP, FP, FN = result

    print("Precision:", round(P, 3))
    print("Recall:", round(R, 3))
    print("F-measure:", round(F, 3))

    return result


def evaluate_dataset(song_dict, chord_recognition_func):

    results = []

    for s in song_dict:

        audio_path = song_dict[s][2]
        annotation_path = song_dict[s][3]

        result = evaluate_song_with_your_method(
            audio_path,
            annotation_path,
            chord_recognition_func
        )

        results.append(result[2])

    print("\n==========================")
    print("FINAL RESULT")
    print("==========================")
    print("Mean F-measure:", round(np.mean(results), 3))


# -------------------------------------------------
# DEFINE DATASET (same as libfmp notebook)
# -------------------------------------------------

song_dict = {}

song_dict[0] = [
    'LetItB',
    'r',
    '../data/C5/FMP_C5_Audio_Beatles_LetItBe_Beatles_1970-LetItBe-06.wav',
    '../data/C5/FMP_C5_Audio_Beatles_LetItBe_Beatles_1970-LetItBe-06_Chords_simplified.csv'
]

song_dict[1] = [
    'HereCo',
    'b',
    '../data/C5/FMP_C5_Audio_Beatles_HereComesTheSun_Beatles_1969-AbbeyRoad-07.wav',
    '../data/C5/FMP_C5_Audio_Beatles_HereComesTheSun_Beatles_1969-AbbeyRoad-07_Chords_simplified.csv'
]

song_dict[2] = [
    'ObLaDi',
    'c',
    '../data/C5/FMP_C5_Audio_Beatles_ObLaDiObLaDa_Beatles_1968-TheBeatlesTheWhiteAlbumDisc1-04.wav',
    '../data/C5/FMP_C5_Audio_Beatles_ObLaDiObLaDa_Beatles_1968-TheBeatlesTheWhiteAlbumDisc1-04_Chords_simplified.csv'
]

song_dict[3] = [
    'PennyL',
    'g',
    '../data/C5/FMP_C5_Audio_Beatles_PennyLane_Beatles_1967-MagicalMysteryTour-09.wav',
    '../data/C5/FMP_C5_Audio_Beatles_PennyLane_Beatles_1967-MagicalMysteryTour-09_Chords_simplified.csv'
]


# -------------------------------------------------
# RUN EVALUATION
# -------------------------------------------------

if __name__ == "__main__":

    evaluate_dataset(
        song_dict,
        chord_recognition
    )
