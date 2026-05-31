from transformers import pipeline

print("XLMR LOADED")

classifier = pipeline(
    "text-classification",
    model="papluca/xlm-roberta-base-language-detection"
)

def predict_xlmr(text):

    print("PREDICT CALLED")

    result = classifier(text)

    return {
        "label": result[0]["label"],
        "score": result[0]["score"]
    }
