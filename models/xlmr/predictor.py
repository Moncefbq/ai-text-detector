from transformers import pipeline

classifier = pipeline(
    "text-classification",
    model="papluca/xlm-roberta-base-language-detection"
)

def predict_xlmr(text):
    result = classifier(text)

    return {
        "label": result[0]["label"],
        "score": result[0]["score"]
    }
