from backend.services.preprocessing import split_sentences
from models.xlmr.predictor import predict_xlmr


def classify_score(score: float) -> str:
    if score >= 0.75:
        return "AI"
    elif score >= 0.45:
        return "MIXED"
    else:
        return "HUMAN"


def analyze_sentences(text: str):
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
            "risk_level": classify_score(score)
        })

    return results
