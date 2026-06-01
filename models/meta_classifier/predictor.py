import joblib
import pandas as pd

model = joblib.load(
    "models/meta_classifier/meta_model.pkl"
)

def predict_meta(features):

    df = pd.DataFrame([features])

    prediction = model.predict(df)[0]

    probability = model.predict_proba(df)[0]

    return {
        "label": prediction,
        "score": max(probability)
    }
