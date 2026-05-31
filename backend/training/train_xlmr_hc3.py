from datasets import load_dataset, Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer
)
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import numpy as np


MODEL_NAME = "xlm-roberta-base"
OUTPUT_DIR = "./models/final_xlmr"

MAX_LENGTH = 256
TRAIN_LIMIT = 2000
TEST_LIMIT = 500


print("Chargement du dataset HC3...")

raw_dataset = load_dataset("Hello-SimpleAI/HC3", "all")


def extract_texts(dataset_split):
    texts = []
    labels = []

    for example in dataset_split:
        human_answers = example.get("human_answers", [])
        chatgpt_answers = example.get("chatgpt_answers", [])

        if human_answers is None:
            human_answers = []

        if chatgpt_answers is None:
            chatgpt_answers = []

        for human_text in human_answers:
            if isinstance(human_text, str) and len(human_text.strip()) > 20:
                texts.append(human_text.strip())
                labels.append(0)

        for ai_text in chatgpt_answers:
            if isinstance(ai_text, str) and len(ai_text.strip()) > 20:
                texts.append(ai_text.strip())
                labels.append(1)

    return Dataset.from_dict({
        "text": texts,
        "label": labels
    })


print("Préparation des textes Human / ChatGPT...")

dataset = extract_texts(raw_dataset["train"])

dataset = dataset.shuffle(seed=42)

dataset = dataset.train_test_split(test_size=0.2, seed=42)

if TRAIN_LIMIT:
    dataset["train"] = dataset["train"].select(
        range(min(TRAIN_LIMIT, len(dataset["train"])))
    )

if TEST_LIMIT:
    dataset["test"] = dataset["test"].select(
        range(min(TEST_LIMIT, len(dataset["test"])))
    )


print("Nombre exemples train :", len(dataset["train"]))
print("Nombre exemples test :", len(dataset["test"]))


print("Chargement tokenizer...")

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)


def tokenize(batch):
    return tokenizer(
        batch["text"],
        truncation=True,
        padding="max_length",
        max_length=MAX_LENGTH
    )


print("Tokenization...")

tokenized_dataset = dataset.map(tokenize, batched=True)

tokenized_dataset = tokenized_dataset.remove_columns(["text"])

tokenized_dataset.set_format("torch")


print("Chargement du modèle XLM-RoBERTa...")

model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=2,
    id2label={
        0: "HUMAN",
        1: "AI"
    },
    label2id={
        "HUMAN": 0,
        "AI": 1
    }
)


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=1)

    return {
        "accuracy": accuracy_score(labels, predictions),
        "precision": precision_score(labels, predictions),
        "recall": recall_score(labels, predictions),
        "f1": f1_score(labels, predictions)
    }


training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    eval_strategy="epoch",
    save_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    num_train_epochs=2,
    weight_decay=0.01,
    logging_dir="./logs",
    logging_steps=50,
    load_best_model_at_end=True,
    save_total_limit=2,
    report_to="none"
)


trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["test"],
    tokenizer=tokenizer,
    compute_metrics=compute_metrics
)


print("Début entraînement...")

trainer.train()


print("Sauvegarde du modèle...")

trainer.save_model(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)


print("Modèle sauvegardé dans :", OUTPUT_DIR)
print("Entraînement terminé.")
