import argparse
import traceback
import os
import soundfile as sf
import tempfile
from reqognition.test.main import compute_aos

from reqognition.test.recognizer import (
    recognize_test_first_20s,
    recognize_microphone
)


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

def test_aos(dataset_path, recognize_file_fn):
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

        est_segments = recognize_file_fn(wav_path)

        if not est_segments:
            print("No chords detected")
            continue

        aos = compute_aos(ref_segments, est_segments)

        print(f"AOS: {aos:.4f}")

        scores.append(aos)

    if not scores:
        return 0.0

    mean_aos = sum(scores) / len(scores)

    print(f"\n=== MEAN AOS: {mean_aos:.4f} ===")

    return mean_aos


# ================= MICROPHONE ADAPTER =================

def recognize_from_audio(y, sr, recognize_file_fn):
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        path = tmp.name

    sf.write(path, y, sr)

    try:
        return recognize_file_fn(path)
    finally:
        os.remove(path)


# ================= MAIN =================

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--file", "-f", help="Path to WAV file")
    parser.add_argument("--test", action="store_true")
    parser.add_argument("--dataset", help="Path to dataset")
    parser.add_argument("--aos", action="store_true")

    parser.add_argument(
        "--type",
        choices=["madmom", "omnizart"],
        default="madmom",
        help="Recognition method"
    )

    args = parser.parse_args()

    # ================= METHOD SELECT =================

    if args.type == "madmom":
        from reqognition.neural.madmom import recognize_madmom_file
        recognize_file_fn = recognize_madmom_file

    elif args.type == "omnizart":
        from reqognition.neural.omny import recognize_omnizart_file
        recognize_file_fn = recognize_omnizart_file

    else:
        raise ValueError("Unknown recognition type")

    print(f"[INFO] recognizer: {args.type}")

    try:

        # ================= AOS =================

        if args.aos:

            if args.file:

                lab_path = os.path.splitext(args.file)[0] + ".lab"

                ref_segments = load_lab_file(lab_path)

                est_segments = recognize_file_fn(args.file)

                aos = compute_aos(ref_segments, est_segments)

                print(f"\n=== FILE AOS: {aos:.4f} ===")

                return

            test_aos(args.dataset, recognize_file_fn)
            return

        # ================= TEST MODE =================

        if args.test:

            chords = recognize_test_first_20s(
                args.dataset,
                recognize_file_fn
            )

            for c in chords or []:
                print(c)

            return

        # ================= FILE MODE =================

        if args.file:

            chords = recognize_file_fn(args.file)

            for c in chords or []:
                print(c)

            return

        # ================= MICROPHONE =================

        def recognition_adapter(y, sr, is_online):
            return recognize_from_audio(y, sr, recognize_file_fn)

        recognize_microphone(recognition_adapter)

    except Exception as e:

        print("\n=== ERROR ===")

        print("TYPE:", type(e))
        print("REPR:", repr(e))

        traceback.print_exc()


if __name__ == "__main__":
    main()