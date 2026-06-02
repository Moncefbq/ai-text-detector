import statistics
from backend.services.preprocessing import split_sentences


def calculate_burstiness(text: str) -> float:
    sentences = split_sentences(text)

    lengths = [len(sentence.split()) for sentence in sentences if len(sentence.split()) > 0]

    if len(lengths) < 2:
        return 0.0

    mean_len = statistics.mean(lengths)
    std_len = statistics.stdev(lengths)

    if mean_len == 0:
        return 0.0

    burstiness = std_len / mean_len

    return round(burstiness, 4)
