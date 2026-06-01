from backend.services.preprocessing import split_sentences
from models.xlmr.predictor import predict_xlmr


def ai_probability(label: str, score: float) -> float:
    if label == "AI":
        return score
    return 1 - score


def classify_score(ai_score: float) -> str:
    if ai_score >= 0.75:
        return "AI"
    elif ai_score >= 0.45:
        return "MIXED"
    else:
        return "HUMAN"


def analyze_sentences(text: str):
    sentences = split_sentences(text)

    results = []

    for sentence in sentences:
        if len(sentence.strip()) < 3:
            continue

        prediction = predict_xlmr(sentence)

        label = prediction["label"]
        score = prediction["score"]

        ai_score = ai_probability(label, score)

        results.append({
            "sentence": sentence,
            "label": label,
            "score": round(score, 4),
            "ai_score": round(ai_score, 4),
            "risk_level": classify_score(ai_score)
        })

    return results
