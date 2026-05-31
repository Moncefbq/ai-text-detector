from transformers import pipeline

classifier = pipeline(
    "text-classification",
    model="./models/final_xlmr",
    tokenizer="./models/final_xlmr"
)

def predict_xlmr(text):
    result = classifier(text)

    return {
        "label": result[0]["label"],
        "score": result[0]["score"]
    }
