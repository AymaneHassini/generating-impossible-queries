from transformers import AutoModelForSequenceClassification, AutoTokenizer, BertForSequenceClassification, BertTokenizerFast, Trainer, TrainingArguments
import torch

checkpoint = "./checkpoint"

device = 'cuda' if torch.cuda.is_available() else 'cpu'

training_args = TrainingArguments(
     "tmp", 
      disable_tqdm=True,
      run_name='generate-impossible-query',
)

def load_classifier_distilbert():

    print(device)
    # Preparing the model
    classifier = AutoModelForSequenceClassification.from_pretrained(
        checkpoint, num_labels=3
    ).to(device)
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    trainer = Trainer(
        classifier,
        args=training_args,
    )

    return trainer, tokenizer


def load_classifier_bert():
    # Preparing the model
    classifier = BertForSequenceClassification.from_pretrained(
      checkpoint,
      num_labels=3,
  )
    tokenizer = BertTokenizerFast.from_pretrained(checkpoint, max_length=512)
    trainer = Trainer(
        classifier,
        args=training_args,
    )

    return trainer, tokenizer