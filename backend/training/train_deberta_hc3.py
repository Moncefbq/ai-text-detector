from datasets import load_dataset, Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import TrainingArguments, Trainer
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import numpy as np

MODEL_NAME = "microsoft/mdeberta-v3-base"
OUTPUT_DIR = "./models/final_deberta"

MAX_LENGTH = 256
TRAIN_LIMIT = 2000
TEST_LIMIT = 500

print("Chargement HC3...")

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


dataset = extract_texts(raw_dataset["train"])
dataset = dataset.shuffle(seed=42)
dataset = dataset.train_test_split(test_size=0.2, seed=42)

dataset["train"] = dataset["train"].select(range(min(TRAIN_LIMIT, len(dataset["train"]))))
dataset["test"] = dataset["test"].select(range(min(TEST_LIMIT, len(dataset["test"]))))

print("Train :", len(dataset["train"]))
print("Test :", len(dataset["test"]))

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)


def tokenize(batch):
    return tokenizer(
        batch["text"],
        truncation=True,
        padding="max_length",
        max_length=MAX_LENGTH
    )


tokenized_dataset = dataset.map(tokenize, batched=True)
tokenized_dataset = tokenized_dataset.remove_columns(["text"])
tokenized_dataset.set_format("torch")

model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=2,
    id2label={0: "HUMAN", 1: "AI"},
    label2id={"HUMAN": 0, "AI": 1}
)


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=1)

    return {
        "accuracy": accuracy_score(labels, preds),
        "precision": precision_score(labels, preds),
        "recall": recall_score(labels, preds),
        "f1": f1_score(labels, preds)
    }


training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    num_train_epochs=2,
    weight_decay=0.01,
    logging_dir="./logs_deberta",
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

print("Début entraînement DeBERTa...")

trainer.train()

trainer.save_model(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)

print("Modèle DeBERTa sauvegardé dans :", OUTPUT_DIR)
