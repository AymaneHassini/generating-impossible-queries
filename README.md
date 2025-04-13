# Generating Impossible Queries

This repository contains code for a deep learning-based pipeline designed to generate "impossible queries" by interpreting inferred relationships and latent attributes that are not directly stored in database schemas. Our system integrates an LLM (Gemini 1.5 Pro) for multimodal reasoning and a fine-tuned BERT model for structured classification. The pipeline enriches static database records with derived features, enabling dynamic query creation.

## Table of Contents

- [Overview](#overview)
- [Folder Structure](#folder-structure)
- [Pipeline Usage](#pipeline-usage)
- [Training BERT](#training-bert)
- [MySQL Database Backup](#mysql-database-backup)
- [Dependencies](#dependencies)
- [License](#license)

## Overview

Our pipeline uses multimodal reasoning to process data from various sources (text, images, documents) and generate actionable insights. The core components include:

- **LLM (Gemini 1.5 Pro):** Extracts latent, context-rich features from multimodal inputs.
- **BERT:** Fine-tuned to classify the derived features, outputting a "Match Score" (0–100) based on how well the record satisfies a given query.

The interactive Jupyter notebook `pipeline_example.ipynb` demonstrates the entire workflow—from connecting to the database to invoking the LLM API and classifying outputs with BERT (guided by specified match guidelines).

## Folder Structure

```
├── benchmark.ipynb
├── bert
│   └── train-bert.py
├── checkpoint
│   ├── config.json
│   ├── model.safetensors
│   ├── special_tokens_map.json
│   ├── tokenizer_config.json
│   ├── tokenizer.json
│   ├── training_args.bin
│   └── vocab.txt
├── database
│   ├── connect.py
│   └── database_backup.sql
├── dataset
│   └── dataset-match-score.csv
├── ground_truth
│   ├── label_blue_dispenser_head.npy
│   ├── label_dispenser_pump.npy
│   └── label_real_baby.npy
├── models
│   ├── classifier.py
│   └── llm.py
├── pipeline_example.ipynb
├── preprocessing
│   ├── document.py
│   ├── image.py
│   └── text.py
├── README.md
└── requirements.txt
```

## Pipeline Usage

- Open `pipeline_example.ipynb` to see a full demonstration of our pipeline in action.  
  This notebook shows how the system connects to a database, processes multimodal inputs using Gemini 1.5 Pro and BERT (guided by a set of defined match guidelines), and generates queries based on enhanced static records.

## Training BERT

To train the BERT classifier on our custom dataset, execute the script `bert/train-bert.py`. The main training routine uses Hugging Face's Trainer with the following key steps:

- Loads data from `dataset/dataset-match-score.csv` and cleans the text.
- Splits the dataset into training and validation sets.
- Tokenizes the text using `BertTokenizerFast` from the specified checkpoint.
- Initializes a BERT model (`bert-base-cased` by default) with a custom classification head for 3 classes (Accept, Recommend, Reject).
- Configures training parameters (e.g., number of epochs, batch size, learning rate, weight decay).
- Computes evaluation metrics (accuracy, precision, recall, F1 score) during training.

Example command-line usage:

```bash
python bert/train-bert.py --data_path ../dataset/dataset-match-score.csv --checkpoint bert-base-cased --batch_size 32 --epochs 2 --learning_rate 5e-4 --weight_decay 5e-4 --output_dir train-checkpoints
```

## MySQL Database Backup

```bash
mysql -u [username] -p [database_name] < database/database_backup.sql
```

## Dependencies

```
torch==2.4.0
transformers==4.45.1
pandas==2.2.2
scikit-learn==1.5.1
numpy==1.26.4
wandb==0.18.3
mysql-connector-python==9.1.0
Pillow==10.4.0
requests==2.32.3
google-generativeai==0.8.3
datasets==3.0.1
tqdm==4.66.5
python-dotenv==1.0.1
```

```bash
pip install -r requirements.txt
```

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
