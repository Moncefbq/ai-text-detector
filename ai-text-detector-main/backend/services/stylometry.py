def lexical_richness(text: str) -> float:
    words = text.lower().split()
    if not words:
        return 0.0
    return len(set(words)) / len(words)

def average_sentence_length(text: str) -> float:
    sentences = [s for s in text.split(".") if s.strip()]
    words = text.split()
    if not sentences:
        return 0.0
    return len(words) / len(sentences)
