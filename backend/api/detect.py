from backend.services.preprocessing import clean_text
from backend.services.language_detection import detect_language
from backend.services.stylometry import lexical_richness, average_sentence_length
from backend.services.sentence_analysis import analyze_sentences
from backend.services.burstiness import calculate_burstiness
from backend.services.perplexity import calculate_perplexity

from models.xlmr.predictor import predict_xlmr
from models.deberta.predictor import predict_deberta


def ai_probability(label: str, score: float) -> float:
    if label == "AI":
        return score
    return 1 - score


def analyze_text(text: str):

    cleaned_text = clean_text(text)
    language = detect_language(cleaned_text)

    lexical_score = lexical_richness(cleaned_text)
    avg_sentence_len = average_sentence_length(cleaned_text)

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

    sentence_results = analyze_sentences(cleaned_text)

    total_sentences = len(sentence_results)
    ai_sentences = [s for s in sentence_results if s["label"] == "AI"]
    mixed_sentences = [s for s in sentence_results if s["risk_level"] == "MIXED"]

    if total_sentences > 0:
        sentence_ai_score = len(ai_sentences) / total_sentences
    else:
        sentence_ai_score = 0.0

    stylometry_ai_score = 1 - lexical_score

    final_ai_score = (
        0.35 * xlmr_ai_score +
        0.35 * deberta_ai_score +
        0.20 * sentence_ai_score +
        0.10 * stylometry_ai_score
    )

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

        "deberta_label": deberta_label,
        "deberta_score": round(deberta_score, 4),
        "deberta_ai_score": round(deberta_ai_score, 4),

        "lexical_richness": round(lexical_score, 4),
        "stylometry_ai_score": round(stylometry_ai_score, 4),
        "average_sentence_length": avg_sentence_len,

        "burstiness": burstiness,
        "perplexity": perplexity,

        "sentence_ai_score": round(sentence_ai_score, 4),
        "suspicious_sentences_count": len(ai_sentences) + len(mixed_sentences),
        "total_sentences": total_sentences,
        "sentence_analysis": sentence_results
    }
