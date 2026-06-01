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
        0.30 * xlmr_ai_score +
        0.30 * deberta_ai_score +
        0.20 * sentence_ai_score +
        0.10 * stylometry_ai_score +
        0.05 * burstiness_score +
        0.05 * perplexity_score
    )

    return round(final_score, 4)


def get_final_label(final_ai_score: float) -> str:
    if final_ai_score >= 0.65:
        return "AI"
    elif final_ai_score <= 0.35:
        return "HUMAN"
    else:
        return "MIXED"


def get_confidence(final_ai_score: float) -> float:
    if final_ai_score >= 0.65:
        return round(final_ai_score, 4)
    elif final_ai_score <= 0.35:
        return round(1 - final_ai_score, 4)
    else:
        return round(1 - abs(0.5 - final_ai_score), 4)
