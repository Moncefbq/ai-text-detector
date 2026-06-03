from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer
)
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import numpy as np

MODEL_NAME = "xlm-roberta-large"

print("Chargement HC3...")

dataset = load_dataset("Hello-SimpleAI/HC3", "all")

human_texts = dataset["train"]["human_answers"][:1000]
ai_texts = dataset["train"]["chatgpt_answers"][:1000]

texts = []
labels = []

for h in human_texts:
    if len(h) > 0:
        texts.append(h[0])
        labels.append(0)

for a in ai_texts:
    if len(a) > 0:
        texts.append(a[0])
        labels.append(1)

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

encodings = tokenizer(
    texts,
    truncation=True,
    padding=True,
    max_length=512
)

class DatasetHC3:
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {
            key: np.array(val[idx])
            for key, val in self.encodings.items()
        }
        item["labels"] = np.array(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

dataset_final = DatasetHC3(encodings, labels)

model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=2
)

def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)

    precision, recall, f1, _ = precision_recall_fscore_support(
        labels,
        preds,
        average="binary"
    )

    acc = accuracy_score(labels, preds)

    return {
        "accuracy": acc,
        "precision": precision,
        "recall": recall,
        "f1": f1
    }

training_args = TrainingArguments(
    output_dir="./models/final_xlmr_large",
    num_train_epochs=2,
    per_device_train_batch_size=1,
    save_strategy="epoch",
    logging_steps=50
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset_final,
    eval_dataset=dataset_final,
    compute_metrics=compute_metrics
)

trainer.train()

print("Sauvegarde...")

trainer.save_model("./models/final_xlmr_large")
tokenizer.save_pretrained("./models/final_xlmr_large")

print("Terminé.")
