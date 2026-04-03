from reqognition.metric.distance.recognition import chord_recognition_rt

def recognize_manual(
    y,
    sr,
    isOnline=False,
    num_harmonics: int = 1,
    measure: str = "KL2",
    filtering: str = "LP+M",
    L: int = 17,
):
    chords, hop_time = chord_recognition_rt(
        y,
        sr,
        isOnline,
        num_harmonics=num_harmonics,
        measure=measure,
        filtering=filtering,
        L=L
    )

    return chords_to_segments(chords, hop_time)

def chords_to_segments(chords, hop_time, min_duration=0.2):
    segments = []

    start = 0.0
    prev = chords[0]

    for i in range(1, len(chords)):
        if chords[i] != prev:
            end = i * hop_time

            if end - start > min_duration:
                segments.append((start, end, prev))

            start = end
            prev = chords[i]

    end = len(chords) * hop_time

    if end - start > min_duration:
        segments.append((start, end, prev))

    return segments