import pandas as pd
from sklearn.model_selection import train_test_split
from transformers import  AutoModelForSequenceClassification, AutoTokenizer
import os
import wandb
from datasets import Dataset, DatasetDict
from sklearn.metrics import classification_report



wandb.init(
    project="impossible-querry-pipeline",
    name="gemini-distilbert"
)

df = pd.read_csv("../dataset/dataset-all.csv")
df['text'] = df['text'].apply(lambda x: x.replace('\n', '').replace('*', ''))

checkpoint = "distilbert-base-cased"
classifier = AutoModelForSequenceClassification.from_pretrained(checkpoint, num_labels=3) 
tokenizer = AutoTokenizer.from_pretrained(checkpoint)

df_train, df_eval = train_test_split(df, train_size=0.8, stratify=df.labels, random_state=42)

raw_datasets = DatasetDict({
    "train": Dataset.from_pandas(df_train),
    "eval": Dataset.from_pandas(df_eval)
})

tokenized_datasets = raw_datasets.map(lambda dataset: tokenizer(dataset['text'], truncation=True), batched=True)
tokenized_datasets = tokenized_datasets.remove_columns(["text", "__index_level_0__"])


from transformers import DataCollatorWithPadding, TrainingArguments, Trainer
import numpy as np
import evaluate
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

training_args = TrainingArguments("train-checkpoints-epoch", 
                                  num_train_epochs=5, 
                                  evaluation_strategy="steps", 
                                  weight_decay=5e-4, 
                                  per_device_train_batch_size=64,
                                  per_device_eval_batch_size=64,
                                  save_strategy="steps",
                                  # fp16=True,
                                  load_best_model_at_end=True,
                                  report_to="wandb",)

def compute_metrics(pred):
    """
    Computes accuracy, F1, precision, and recall for a given set of predictions.

    Args:
        pred (obj): An object containing label_ids and predictions attributes.
            - label_ids (array-like): A 1D array of true class labels.
            - predictions (array-like): A 2D array where each row represents
              an observation, and each column represents the probability of
              that observation belonging to a certain class.

    Returns:
        dict: A dictionary containing the following metrics:
            - Accuracy (float): The proportion of correctly classified instances.
            - F1 (float): The macro F1 score, which is the harmonic mean of precision
              and recall. Macro averaging calculates the metric independently for
              each class and then takes the average.
            - Precision (float): The macro precision, which is the number of true
              positives divided by the sum of true positives and false positives.
            - Recall (float): The macro recall, which is the number of true positives
              divided by the sum of true positives and false negatives.
    """
    labels = pred.label_ids

    preds = pred.predictions.argmax(-1)

    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='macro')

    acc = accuracy_score(labels, preds)


    return {
        'Accuracy': acc,
        'F1': f1,
        'Precision': precision,
        'Recall': recall
    }


trainer = Trainer(
    classifier,
    training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["eval"],
    data_collator=data_collator,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics
)

trainer.train()

trainer.save_model("train-checkpoints-step/best-model/")


y_pred = trainer.predict(tokenized_datasets["eval"]).predictions
y_pred = np.argmax(y_pred, axis=-1)

y_true = tokenized_datasets["eval"]["labels"]
y_true = np.array(y_true)

print(classification_report(y_true, y_pred, digits=4))