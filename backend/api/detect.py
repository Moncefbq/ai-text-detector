from backend.services.preprocessing import clean_text
from backend.services.language_detection import detect_language
from backend.services.stylometry import lexical_richness, average_sentence_length
from backend.services.ensemble import weighted_score

from models.xlmr.predictor import predict_xlmr


def analyze_text(text: str):

    cleaned_text = clean_text(text)

    language = detect_language(cleaned_text)

    lexical_score = lexical_richness(cleaned_text)

    avg_sentence_len = average_sentence_length(cleaned_text)

    prediction = predict_xlmr(cleaned_text)

    xlmr_score = prediction["score"]

    stylometry_score = lexical_score

    final_score = weighted_score(
        xlmr_score,
        stylometry_score
    )

    return {
        "language": language,
        "xlmr_score": xlmr_score,
        "lexical_richness": lexical_score,
        "average_sentence_length": avg_sentence_len,
        "final_ai_score": final_score
    }
