from transformers import pipeline

classifier = pipeline(
    "text-classification",
    model="./models/final_xlmr_large",
    tokenizer="./models/final_xlmr_large",
    truncation=True,
    max_length=512
)

def predict_xlmr_large(text):
    result = classifier(text)[0]

    label = result["label"]
    score = result["score"]

    ai_score = score if label == "LABEL_1" else 1 - score

    return {
        "label": "AI" if ai_score >= 0.5 else "HUMAN",
        "score": round(max(ai_score, 1 - ai_score), 4),
        "ai_score": round(ai_score, 4)
    }
