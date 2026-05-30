def lexical_richness(text):
    words = text.split()

    if len(words) == 0:
        return 0

    return len(set(words)) / len(words)


def average_sentence_length(text):
    sentences = text.split(".")

    if len(sentences) == 0:
        return 0

    words = len(text.split())

    return words / max(len(sentences), 1)
