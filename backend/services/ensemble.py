def normalize_perplexity(perplexity: float) -> float:
    if perplexity <= 0:
        return 0.5

    if perplexity < 20:
        return 0.85
    elif perplexity < 50:
        return 0.65
    elif perplexity < 100:
        return 0.45
    else:
        return 0.25


def normalize_burstiness(burstiness: float) -> float:
    if burstiness < 0.2:
        return 0.85
    elif burstiness < 0.5:
        return 0.65
    elif burstiness < 0.9:
        return 0.45
    else:
        return 0.25


def weighted_score(
    xlmr_ai_score: float,
    deberta_ai_score: float,
    sentence_ai_score: float,
    stylometry_ai_score: float,
    burstiness_score: float = 0.5,
    perplexity_score: float = 0.5
) -> float:

    final_score = (
        0.20 * xlmr_ai_score +
        0.20 * deberta_ai_score +
        0.40 * sentence_ai_score +
        0.10 * stylometry_ai_score +
        0.05 * burstiness_score +
        0.05 * perplexity_score
    )

    if sentence_ai_score == 0 and xlmr_ai_score > 0.90 and deberta_ai_score > 0.90:
        final_score = min(final_score, 0.35)

    return round(final_score, 4)
