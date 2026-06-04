print("🔥 XLMR LARGE VERSION LOADED 🔥")
from backend.services.preprocessing import clean_text
from backend.services.language_detection import detect_language
from backend.services.stylometry import stylometry_report
from backend.services.sentence_analysis import analyze_sentences
from backend.services.burstiness import calculate_burstiness
from backend.services.perplexity import calculate_perplexity

from models.xlmr.predictor import predict_xlmr
from models.deberta.predictor import predict_deberta
from models.xlmr_large.predictor import predict_xlmr_large


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

    xlmr_large_prediction = predict_xlmr_large(cleaned_text)
    xlmr_large_label = xlmr_large_prediction["label"]
    xlmr_large_score = xlmr_large_prediction["score"]
    xlmr_large_ai_score = xlmr_large_prediction["ai_score"]

    deberta_prediction = predict_deberta(cleaned_text)
    deberta_label = deberta_prediction["label"]
    deberta_score = deberta_prediction["score"]
    deberta_ai_score = ai_probability(deberta_label, deberta_score)

    burstiness = calculate_burstiness(cleaned_text)
    perplexity = calculate_perplexity(cleaned_text)

    sentence_results = analyze_sentences(cleaned_text)

    total_sentences = len(sentence_results)
    ai_sentences = [s for s in sentence_results if s["label"] == "AI"]
    mixed_sentences = [s for s in sentence_results if s["risk_level"] == "MIXED"]

    sentence_ai_score = len(ai_sentences) / total_sentences if total_sentences > 0 else 0.0
    stylometry_ai_score = 1 - lexical_score

    final_ai_score = (
        0.25 * xlmr_ai_score +
        0.25 * xlmr_large_ai_score +
        0.25 * deberta_ai_score +
        0.15 * sentence_ai_score +
        0.10 * stylometry_ai_score
    )

    if xlmr_ai_score < 0.10 and xlmr_large_ai_score < 0.10 and sentence_ai_score == 0:
        final_ai_score = min(final_ai_score, 0.08)

    final_ai_score = round(final_ai_score, 4)

    if final_ai_score >= 0.65:
        final_label = "AI"
        confidence = final_ai_score
    elif final_ai_score <= 0.35:
        final_label = "HUMAN"
        confidence = 1 - final_ai_score
    else:
        final_label = "MIXED"
        confidence = 1 - abs(0.5 - final_ai_score)

    return {
        "language": language,

        "final_label": final_label,
        "final_ai_score": final_ai_score,
        "confidence": round(confidence, 4),

        "xlmr_label": xlmr_label,
        "xlmr_score": round(xlmr_score, 4),
        "xlmr_ai_score": round(xlmr_ai_score, 4),

        "xlmr_large_label": xlmr_large_label,
        "xlmr_large_score": round(xlmr_large_score, 4),
        "xlmr_large_ai_score": round(xlmr_large_ai_score, 4),

        "deberta_label": deberta_label,
        "deberta_score": round(deberta_score, 4),
        "deberta_ai_score": round(deberta_ai_score, 4),

        "lexical_richness": round(lexical_score, 4),
        "stylometry_ai_score": round(stylometry_ai_score, 4),
        "average_sentence_length": avg_sentence_len,

        "stylometry": style,

        "burstiness": burstiness,
        "perplexity": perplexity,

        "sentence_ai_score": round(sentence_ai_score, 4),
        "suspicious_sentences_count": len(ai_sentences) + len(mixed_sentences),
        "total_sentences": total_sentences,
        "sentence_analysis": sentence_results
    }
