import numpy as np

NOTE_TO_PITCH_CLASS = {
    "C": 0, "C#": 1, "D": 2, "D#": 3, "E": 4,
    "F": 5, "F#": 6, "G": 7, "G#": 8,
    "A": 9, "A#": 10, "B": 11
}

CHORD_INTERVALS = {
    # Трезвучия
    "maj":  [0, 4, 7],   # мажорное
    "min":  [0, 3, 7],   # минорное
    # # "dim":  [0, 3, 6],   # уменьшённое
    # # "aug":  [0, 4, 8],   # увеличенное
    #
    # # Септаккорды
    #
    # # 1. Большой мажорный септаккорд (бБ7) — major seventh
    # "maj7": [0, 4, 7, 11],
    # #
    # # # 2. Большой минорный септаккорд (бМ7)
    # # # часто называется min(maj7)
    # # "minmaj7": [0, 3, 7, 11],
    # #
    # # # 3. Большой увеличенный септаккорд (бУв7)
    # # "augmaj7": [0, 4, 8, 11],
    # #
    # # # 4. Малый мажорный септаккорд (мБ7) — доминантсептаккорд
    # "7": [0, 4, 7, 10],
    #
    # # 5. Малый минорный септаккорд (мМ7)
    # "min7": [0, 3, 7, 10],
    #
    # # 6. Малый уменьшённый септаккорд (мУм7) — half-diminished
    # "m7b5": [0, 3, 6, 10],
    #
    # # 7. Уменьшённый септаккорд (Ум7)
    # "dim7": [0, 3, 6, 9],
}

def generate_all_templates(num_harmonics=4):
    templates = {}
    for root, root_pc in NOTE_TO_PITCH_CLASS.items():
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

    if template.sum() > 0:
        template /= template.sum()

    return template