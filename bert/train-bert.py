import torch
import pandas as pd
from transformers import BertForSequenceClassification, BertTokenizerFast, Trainer, TrainingArguments
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import os
import wandb
import argparse


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default="../dataset/dataset-macth-score.csv")
    parser.add_argument("--checkpoint", type=str, default="bert-base-cased")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--learning_rate", type=float, default=5e-4)
    parser.add_argument("--weight_decay", type=float, default=5e-4)
    parser.add_argument("--output_dir", type=str, default="train-checkpoints")
    return parser

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

class CustomDataset(Dataset):
    """
    Custom Dataset class for handling tokenized text data and corresponding labels.
    Inherits from torch.utils.data.Dataset.
    """
    def __init__(self, encodings, labels):
        """
        Initializes the DataLoader class with encodings and labels.

        Args:
            encodings (dict): A dictionary containing tokenized input text data
                              (e.g., 'input_ids', 'token_type_ids', 'attention_mask').
            labels (list): A list of integer labels for the input text data.
        """
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        """
        Returns a dictionary containing tokenized data and the corresponding label for a given index.

        Args:
            idx (int): The index of the data item to retrieve.

        Returns:
            item (dict): A dictionary containing the tokenized data and the corresponding label.
        """
        item = {key: torch.tensor(val[idx]).clone().detach() for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels.iloc[idx]).clone().detach()
        return item

    def __len__(self):
        """
        Returns the number of data items in the dataset.

        Returns:
            (int): The number of data items in the dataset.
        """
        return len(self.labels)


if __name__ == "__main__":
    wandb.init(
        project="impossible-querry-pipeline",
        name="gemini-bert"
    )

    device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
    print(device)

    args = get_parser().parse_args()

    checkpoint = args.checkpoint
    data_path = args.data_path
    batch_size = args.batch_size
    epochs = args.epochs
    learning_rate = args.learning_rate
    weight_decay = args.weight_decay
    output_dir = args.output_dir

    print(f"Loading data from {data_path}")
    df = pd.read_csv(data_path)
    df['text'] = df['text'].apply(lambda x: x.replace('\n', ' ').replace('*', '')) 


    labels = df['labels'].unique().tolist()
    labels.sort()
    id2label = {i: label for i, label in enumerate(labels)}
    label2id = {label: i for i, label in enumerate(labels)}


    train_texts, val_texts, train_labels, val_labels = train_test_split(
        df['text'], df['labels'], test_size=0.2
    )

    print(f"Train size: {len(train_texts)}")
    print(f"Val size: {len(val_texts)}")
    tokenizer = BertTokenizerFast.from_pretrained(checkpoint, max_length=512)
    train_encodings = tokenizer(list(train_texts), truncation=True, padding=True, return_tensors="pt")
    val_encodings = tokenizer(list(val_texts), truncation=True, padding=True, return_tensors="pt")



    train_dataset = CustomDataset(train_encodings, train_labels)
    val_dataset = CustomDataset(val_encodings, val_labels)

    num_labels = len(label2id)
    model = BertForSequenceClassification.from_pretrained(
        checkpoint,
        num_labels=num_labels,
        id2label=id2label,
        label2id=label2id
    ).to(device)

    training_args = TrainingArguments(
        output_dir, 
        num_train_epochs=epochs, 
        eval_strategy="steps", 
        weight_decay=weight_decay, 
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        save_strategy="steps",
        logging_steps=50,
        load_best_model_at_end=True,
        #   report_to="wandb",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics= compute_metrics
    )

    trainer.train()

    tokenizer.save_pretrained(os.path.join(output_dir, "best-model"))
    trainer.save_model(os.path.join(output_dir, "best-model"))