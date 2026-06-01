from transformers import pipeline

classifier = pipeline(
    "text-classification",
    model="./models/final_deberta",
    tokenizer="./models/final_deberta"
)


def predict_deberta(text):
    result = classifier(text, truncation=True, max_length=512)

    return {
        "label": result[0]["label"],
        "score": result[0]["score"]
    }
