import numpy as np

NOTE_TO_PC = {
    "C": 0, "C#": 1, "D": 2, "D#": 3, "E": 4,
    "F": 5, "F#": 6, "G": 7, "G#": 8,
    "A": 9, "A#": 10, "B": 11
}

CHORD_INTERVALS = {
    "maj": [0, 4, 7],
    "min": [0, 3, 7],
    "7": [0, 4, 7, 10]
}


def generate_all_templates(num_harmonics=4):
    templates = {}
    for root, root_pc in NOTE_TO_PC.items():
        for chord_type, intervals in CHORD_INTERVALS.items():
            name = f"{root}:{chord_type}"
            templates[name] = build_chord_template(
                root_pc,
                intervals,
                num_harmonics=num_harmonics
            )
    return templates


def build_chord_template(
    root_pc,
    intervals,
    num_harmonics=1,
    decay=0.6
):
    template = np.zeros(12)

    for interval in intervals:
        note_pc = (root_pc + interval) % 12

        for i in range(1, num_harmonics + 1):
            amp = decay ** (i - 1)
            harmonic_pc = (note_pc + round(12 * np.log2(i))) % 12
            template[harmonic_pc] += amp

    # Normalize so sum = 1
    if template.sum() > 0:
        template /= template.sum()

    return template