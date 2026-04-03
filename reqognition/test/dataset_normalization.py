def normalize_chord(label: str):

    if label == "N":
        return "N"

    # убрать бас / инверсии
    label = label.split("/")[0]

    # root
    root = label.split(":")[0]

    # тип аккорда
    if ":min" in label:
        return f"{root}:min"

    if ":maj" in label:
        return f"{root}:maj"

    # доминант, maj7 → major
    if ":7" in label or ":maj7" in label:
        return f"{root}:maj"

    # min7 → minor
    if ":min7" in label:
        return f"{root}:min"

    # powerchord
    if ":5" in label:
        return f"{root}:maj"

    # hdim7 (m7b5)
    if ":hdim7" in label:
        return f"{root}:min"

    return f"{root}:maj"