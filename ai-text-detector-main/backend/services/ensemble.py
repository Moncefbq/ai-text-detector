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


def weighted_score(xlmr_score, stylometry_score, burstiness_score=0.5, perplexity_score=0.5):
    final_score = (
        0.60 * xlmr_score +
        0.15 * stylometry_score +
        0.10 * burstiness_score +
        0.15 * perplexity_score
    )

    return round(final_score, 4)
