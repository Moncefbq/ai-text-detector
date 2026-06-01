from backend.services.preprocessing import clean_text
from backend.services.language_detection import detect_language
from backend.services.stylometry import stylometry_report
from backend.services.sentence_analysis import analyze_sentences
from backend.services.burstiness import calculate_burstiness
from backend.services.perplexity import calculate_perplexity

from backend.services.ensemble import (
    weighted_score,
    normalize_perplexity,
    normalize_burstiness,
    get_final_label,
    get_confidence
)

from models.xlmr.predictor import predict_xlmr
from models.deberta.predictor import predict_deberta


def ai_probability(label: str, score: float) -> float:
    if label == "AI":
        return score
    return 1 - score


def analyze_text(text: str):

    cleaned_text = clean_text(text)
    language = detect_language(cleaned_text)

    style = stylometry_report(cleaned_text)

    lexical_score = style["lexical_richness"]
    avg_sentence_len = style["average_sentence_length"]

    xlmr_prediction = predict_xlmr(cleaned_text)
    xlmr_label = xlmr_prediction["label"]
    xlmr_score = xlmr_prediction["score"]
    xlmr_ai_score = ai_probability(xlmr_label, xlmr_score)

    deberta_prediction = predict_deberta(cleaned_text)
    deberta_label = deberta_prediction["label"]
    deberta_score = deberta_prediction["score"]
    deberta_ai_score = ai_probability(deberta_label, deberta_score)

    burstiness = calculate_burstiness(cleaned_text)
    perplexity = calculate_perplexity(cleaned_text)

    burstiness_score = normalize_burstiness(burstiness)
    perplexity_score = normalize_perplexity(perplexity)

    sentence_results = analyze_sentences(cleaned_text)

    total_sentences = len(sentence_results)
    ai_sentences = [s for s in sentence_results if s["label"] == "AI"]
    mixed_sentences = [s for s in sentence_results if s["risk_level"] == "MIXED"]

    if total_sentences > 0:
        sentence_ai_score = len(ai_sentences) / total_sentences
    else:
        sentence_ai_score = 0.0

    stylometry_ai_score = 1 - lexical_score

    final_ai_score = weighted_score(
        xlmr_ai_score=xlmr_ai_score,
        deberta_ai_score=deberta_ai_score,
        sentence_ai_score=sentence_ai_score,
        stylometry_ai_score=stylometry_ai_score,
        burstiness_score=burstiness_score,
        perplexity_score=perplexity_score
    )

    final_label = get_final_label(final_ai_score)
    confidence = get_confidence(final_ai_score)

    return {
        "language": language,

        "final_label": final_label,
        "final_ai_score": final_ai_score,
        "confidence": confidence,

        "xlmr_label": xlmr_label,
        "xlmr_score": round(xlmr_score, 4),
        "xlmr_ai_score": round(xlmr_ai_score, 4),

        "deberta_label": deberta_label,
        "deberta_score": round(deberta_score, 4),
        "deberta_ai_score": round(deberta_ai_score, 4),

        "lexical_richness": round(lexical_score, 4),
        "stylometry_ai_score": round(stylometry_ai_score, 4),
        "average_sentence_length": avg_sentence_len,
        "stylometry": style,

        "burstiness": burstiness,
        "burstiness_score": burstiness_score,
        "perplexity": perplexity,
        "perplexity_score": perplexity_score,

        "sentence_ai_score": round(sentence_ai_score, 4),
        "suspicious_sentences_count": len(ai_sentences) + len(mixed_sentences),
        "total_sentences": total_sentences,
        "sentence_analysis": sentence_results
    }
