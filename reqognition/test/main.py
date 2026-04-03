import argparse
import traceback
import os
import librosa
from reqognition.test.dataset_normalization import normalize_chord
from reqognition.test.recognizer import (
    recognize_file,
    recognize_test_first_20s,
    recognize_microphone
)

from reqognition.metric.manual_req import recognize_manual
from reqognition.metric.librosa_lib import recognize_fmp
from reqognition.probabilistic.librosa_lib import recognize_hmm
from reqognition.probabilistic.bello_gauss_hmm import recognize_hmm_manual
import glob
from reqognition.probabilistic.bello_gauss_hmm import train_hmm_manual
from reqognition.probabilistic.template_hmm import recognize_hmm as template_hmm


# ================= AOS =================

def load_lab_file(path):
    segments = []
    with open(path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) != 3:
                continue
            start, end, label = parts
            segments.append((float(start), float(end), label))
    return segments


def compute_aos(ref_segments, est_segments):
    print("Computing Chords")
    for c in est_segments or []:
        print(c)

    total = 0.0
    correct = 0.0

    i = j = 0

    while i < len(ref_segments) and j < len(est_segments):

        rs, re, rl = ref_segments[i]
        es, ee, el = est_segments[j]

        rl = normalize_chord(rl)
        el = normalize_chord(el)

        start = max(rs, es)
        end = min(re, ee)

        if end > start:
            overlap = end - start
            total += overlap
            if rl == el:
                correct += overlap

        if re < ee:
            i += 1
        else:
            j += 1

    if total == 0:
        return 0.0

    return correct / total

def test_aos(dataset_path, recognition_fn):

    wav_files = [f for f in os.listdir(dataset_path) if f.endswith(".wav")]

    if not wav_files:
        raise RuntimeError("No WAV files found")

    lab_path = os.path.join(dataset_path, "guitar_annotation.lab")

    if not os.path.exists(lab_path):
        raise RuntimeError("guitar_annotation.lab not found")

    ref_segments = load_lab_file(lab_path)

    scores = []

    for wav_file in wav_files:

        wav_path = os.path.join(dataset_path, wav_file)

        print(f"Testing: {wav_file}")

        y, sr = librosa.load(wav_path, sr=None, mono=True)
        y = y[:int(240 * sr)]

        est_segments = recognition_fn(y, sr)

        if not est_segments:
            print("No chords detected")
            continue

        aos = compute_aos(ref_segments, est_segments)

        print(f"AOS: {aos:.4f}")
        scores.append(aos)

    if not scores:
        print("No valid files evaluated")
        return 0.0

    mean_aos = sum(scores) / len(scores)

    print(f"\n=== MEAN AOS: {mean_aos:.4f} ===")

    return mean_aos

# ================= MAIN =================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", "-f", help="Path to WAV file")
    parser.add_argument("--test", action="store_true")
    parser.add_argument("--dataset", help="Path to dataset")
    parser.add_argument("--aos", action="store_true")
    parser.add_argument(
        "--type",
        choices=["dist", "lib_dist", "lib_hmm", "t_hmm", "gauss_hmm"],
        default="dist",
        help="Recognition method"
    )

    args = parser.parse_args()

    print("=== DEBUG ===")
    print("args:", args)
    print("=============")

    if args.type == "dist":
        recognition_fn = recognize_manual
    elif args.type == "lib_dist":
        recognition_fn = recognize_fmp
    elif args.type == "lib_hmm":
        recognition_fn = recognize_hmm
    elif args.type == "t_hmm":
        recognition_fn = template_hmm
    elif args.type == "gauss_hmm":
        BASE_DIR = os.path.dirname(os.path.abspath(__file__))

        base_dir = os.path.join(BASE_DIR, "..", "..", "data_sets", "IDMT-SMT-CHORDS", "guitar")

        wav_files = glob.glob(os.path.join(base_dir, "**", "*.wav"), recursive=True)

        train_hmm_manual(wav_files)
        recognition_fn = recognize_hmm_manual
    else:
        raise ValueError("Unknown recognition type")

    print(f"[DEBUG] recognition_fn: {recognition_fn.__name__}")

    try:
        if args.aos:
            if args.file:

                lab_path = os.path.splitext(args.file)[0] + ".lab"

                if not os.path.exists(lab_path):
                    raise RuntimeError(f"{lab_path} not found")

                ref_segments = load_lab_file(lab_path)

                y, sr = librosa.load(args.file, sr=None, mono=True)
                y = y[:int(240 * sr)]

                est_segments = recognition_fn(y, sr)

                aos = compute_aos(ref_segments, est_segments)

                print(f"\n=== FILE AOS: {aos:.4f} ===")
                return

            if not args.dataset:
                raise ValueError("--dataset required for dataset AOS test")

            test_aos(args.dataset, recognition_fn)
            return

        if args.test:
            if not args.dataset:
                raise ValueError("--dataset is required in test mode")

            chords = recognize_test_first_20s(
                args.dataset,
                recognition_fn,
            )

            for c in chords or []:
                print(c)
            return

        if args.file:
            chords = recognize_file(
                args.file,
                recognition_fn,
            )

            for c in chords or []:
                print(c)
            return

        recognize_microphone(recognition_fn)

    except Exception as e:
        print("\n=== ERROR ===")
        print(e)
        traceback.print_exc()

if __name__ == "__main__":
    main()