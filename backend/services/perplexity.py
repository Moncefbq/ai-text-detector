import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


MODEL_NAME = "distilgpt2"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)

model.eval()


def calculate_perplexity(text: str) -> float:
    if not text or len(text.split()) < 5:
        return 0.0

    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=512
    )

    with torch.no_grad():
        outputs = model(
            **inputs,
            labels=inputs["input_ids"]
        )

    loss = outputs.loss
    perplexity = torch.exp(loss)

    return round(float(perplexity), 4)
