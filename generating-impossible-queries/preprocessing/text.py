from datasets import Dataset



def tokenize(text, tokenizer):
    dataset = tokenizer(
        text,
        truncation=True,
        padding="max_length",  
        max_length=512,  
        return_tensors="pt",  
    )

    return Dataset.from_dict(
        {"input_ids": dataset["input_ids"], "attention_mask": dataset["attention_mask"]}
    )
