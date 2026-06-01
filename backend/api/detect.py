from backend.services.preprocessing import clean_text
from backend.services.language_detection import detect_language
from backend.services.stylometry import stylometry_report
from backend.services.sentence_analysis import analyze_sentences
from backend.services.burstiness import calculate_burstiness
from backend.services.perplexity import calculate_perplexity

from backend.services.ensemble import (
    normalize_perplexity,
    normalize_burstiness
)

from models.xlmr.predictor import predict_xlmr
from models.deberta.predictor import predict_deberta
#from models.meta_classifier.predictor import predict_meta


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
    ai_sentences = [s for s in sentence_results if s["risk_level"] == "AI"]
    mixed_sentences = [s for s in sentence_results if s["risk_level"] == "MIXED"]

    if total_sentences > 0:
        sentence_ai_score = (
            len(ai_sentences) + 0.5 * len(mixed_sentences)
        ) / total_sentences
    else:
        sentence_ai_score = 0.0

    stylometry_ai_score = 1 - lexical_score

    meta_features = {
        "xlmr_ai_score": round(xlmr_ai_score, 4),
        "deberta_ai_score": round(deberta_ai_score, 4),
        "sentence_ai_score": round(sentence_ai_score, 4),
        "stylometry_ai_score": round(stylometry_ai_score, 4),
        "lexical_richness": round(lexical_score, 4),
        "average_sentence_length": round(avg_sentence_len, 4),
        "burstiness": round(burstiness, 4),
        "burstiness_score": round(burstiness_score, 4),
        "perplexity": round(perplexity, 4),
        "perplexity_score": round(perplexity_score, 4)
    }

    meta_prediction = predict_meta(meta_features)

    final_label = meta_prediction["label"]
    confidence = round(meta_prediction["score"], 4)

    if final_label == "AI":
        final_ai_score = confidence
    elif final_label == "HUMAN":
        final_ai_score = round(1 - confidence, 4)
    else:
        final_ai_score = 0.5

    return {
        "language": language,

        "final_label": final_label,
        "final_ai_score": final_ai_score,
        "confidence": confidence,

        "meta_label": final_label,
        "meta_score": confidence,
        "meta_features": meta_features,

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
