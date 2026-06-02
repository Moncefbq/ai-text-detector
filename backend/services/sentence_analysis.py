from backend.services.preprocessing import split_sentences
from models.xlmr.predictor import predict_xlmr


def classify_score(label: str, score: float) -> str:
    """
    Détermine le niveau de risque réel
    en tenant compte du label prédit.
    """

    if label == "AI":
        if score >= 0.75:
            return "AI"
        elif score >= 0.45:
            return "MIXED"
        else:
            return "HUMAN"

    if label == "HUMAN":
        if score >= 0.75:
            return "HUMAN"
        elif score >= 0.45:
            return "MIXED"
        else:
            return "AI"

    return "MIXED"


def analyze_sentences(text: str):
    """
    Analyse phrase par phrase avec XLM-RoBERTa
    """

    sentences = split_sentences(text)

    results = []

    for sentence in sentences:

        if len(sentence.split()) < 3:
            continue

        prediction = predict_xlmr(sentence)

        score = prediction["score"]
        label = prediction["label"]

        results.append({
            "sentence": sentence,
            "label": label,
            "score": round(score, 4),
            "risk_level": classify_score(label, score)
        })

    return results
