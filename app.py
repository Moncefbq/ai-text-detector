import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

MODEL_PATH = "models/xlmr_detector"

@st.cache_resource
def load_model():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
    model.eval()
    return tokenizer, model

tokenizer, model = load_model()

st.title("AICheck – AI Text Detector")

text = st.text_area("Enter text")

if st.button("Analyze"):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=256)
    with torch.no_grad():
        logits = model(**inputs).logits
    probs = torch.softmax(logits, dim=-1)[0]

    st.metric("AI probability", f"{probs[1]*100:.2f}%")
    st.metric("Human probability", f"{probs[0]*100:.2f}%")

