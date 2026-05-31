from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import TrainingArguments, Trainer
from sklearn.metrics import accuracy_score, f1_score
import numpy as np

MODEL_NAME = "xlm-roberta-base"

# 1. Charger HC3
dataset = load_dataset("Hello-SimpleAI/HC3", "all")

# 2. Préparer les textes
def build_examples(example):
    texts = []
    labels = []

    for human_text in example["human_answers"]:
        if human_text:
            texts.append(human_text)
            labels.append(0)

    for ai_text in example["chatgpt_answers"]:
        if ai_text:
            texts.append(ai_text)
            labels.append(1)

    return {"text": texts, "label": labels}

dataset = dataset["train"].map(
    build_examples,
    batched=False,
    remove_columns=dataset["train"].column_names
)

dataset = dataset.train_test_split(test_size=0.2, seed=42)

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

def tokenize(batch):
    return tokenizer(
        batch["text"],
        truncation=True,
        padding="max_length",
        max_length=256
    )

tokenized = dataset.map(tokenize, batched=True)

model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=2
)

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=1)

    return {
        "accuracy": accuracy_score(labels, preds),
        "f1": f1_score(labels, preds)
    }

training_args = TrainingArguments(
    output_dir="./models/final_xlmr",
    eval_strategy="epoch",
    save_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    num_train_epochs=2,
    weight_decay=0.01,
    logging_dir="./logs",
    load_best_model_at_end=True
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized["train"],
    eval_dataset=tokenized["test"],
    tokenizer=tokenizer,
    compute_metrics=compute_metrics
)

trainer.train()

trainer.save_model("./models/final_xlmr")
tokenizer.save_pretrained("./models/final_xlmr")

print("Modèle sauvegardé dans ./models/final_xlmr")
