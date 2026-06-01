import re
import math
from collections import Counter


def get_words(text: str):
    return re.findall(r"\b\w+\b", text.lower())


def get_sentences(text: str):
    return [s.strip() for s in re.split(r"[.!?]+", text) if s.strip()]


def lexical_richness(text: str) -> float:
    words = get_words(text)
    return round(len(set(words)) / len(words), 4) if words else 0.0


def average_sentence_length(text: str) -> float:
    sentences = get_sentences(text)
    words = get_words(text)
    return round(len(words) / len(sentences), 2) if sentences else 0.0


def word_count(text: str) -> int:
    return len(get_words(text))


def sentence_count(text: str) -> int:
    return len(get_sentences(text))


def average_word_length(text: str) -> float:
    words = get_words(text)
    return round(sum(len(w) for w in words) / len(words), 2) if words else 0.0


def repetition_rate(text: str) -> float:
    words = get_words(text)
    if not words:
        return 0.0

    counts = Counter(words)
    repeated_words = sum(1 for _, count in counts.items() if count > 1)

    return round(repeated_words / len(set(words)), 4) if set(words) else 0.0


def punctuation_density(text: str) -> float:
    if not text:
        return 0.0

    punctuation_count = len(re.findall(r"[.,;:!?]", text))
    return round(punctuation_count / len(text), 4)


def entropy_score(text: str) -> float:
    words = get_words(text)
    if not words:
        return 0.0

    counts = Counter(words)
    total = len(words)

    entropy = 0.0
    for count in counts.values():
        probability = count / total
        entropy -= probability * math.log2(probability)

    return round(entropy, 4)


def hapax_ratio(text: str) -> float:
    words = get_words(text)
    if not words:
        return 0.0

    counts = Counter(words)
    hapax = sum(1 for count in counts.values() if count == 1)

    return round(hapax / len(words), 4)


def sentence_length_variance(text: str) -> float:
    sentences = get_sentences(text)

    if len(sentences) <= 1:
        return 0.0

    lengths = [len(get_words(sentence)) for sentence in sentences]
    mean = sum(lengths) / len(lengths)

    variance = sum((length - mean) ** 2 for length in lengths) / len(lengths)

    return round(variance, 4)


def word_length_variance(text: str) -> float:
    words = get_words(text)

    if len(words) <= 1:
        return 0.0

    lengths = [len(word) for word in words]
    mean = sum(lengths) / len(lengths)

    variance = sum((length - mean) ** 2 for length in lengths) / len(lengths)

    return round(variance, 4)


def stylometry_report(text: str):
    return {
        "word_count": word_count(text),
        "sentence_count": sentence_count(text),
        "lexical_richness": lexical_richness(text),
        "average_sentence_length": average_sentence_length(text),
        "average_word_length": average_word_length(text),
        "repetition_rate": repetition_rate(text),
        "punctuation_density": punctuation_density(text),
        "entropy": entropy_score(text),
        "hapax_ratio": hapax_ratio(text),
        "sentence_length_variance": sentence_length_variance(text),
        "word_length_variance": word_length_variance(text)
    }
