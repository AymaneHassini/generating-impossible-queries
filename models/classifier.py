from transformers import AutoModelForSequenceClassification, AutoTokenizer
from transformers import Trainer

checkpoint = "./checkpoint"


def load_classifier():
    # Preparing the model
    classifier = AutoModelForSequenceClassification.from_pretrained(
        checkpoint, num_labels=3
    )
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    trainer = Trainer(
        classifier,
    )

    return trainer, tokenizer
