# Generating Impossible Queries

[![SSRN Preprint](https://img.shields.io/badge/SSRN-Preprint-b31b1b)](https://ssrn.com/abstract=5071808)  
![Pipeline Architecture](docs/images/cas-sysarch.pdf)

## Features

```python
FEATURES = [
    "Multimodal query processing (text + images + documents)",
    "BERT/Gemini integration for complex reasoning",
    "Three-class confidence scoring (Accept/Recommend/Deny)",
    "Dynamic SQL query generation", 
    "Relational database integration",
]

⚙️ Installation
# Clone repository
git clone https://github.com/AymaneHassini/generating-impossible-queries.git
cd generating-impossible-queries

# Install dependencies
pip install -r requirements.txt

# Configure environment
cp .env.example .env  # Update with your API keys and DB credentials

🗃️ Database Setup
# Create MySQL database
mysql -u root -p -e "CREATE DATABASE prods"

# Import schema and data
mysql -u root -p prods < database/dump.sql

# Verify installation
mysql -u root -p -e "USE prods; SHOW TABLES;"

# Model Training Guide  
💻 **Training Models**  

## BERT Fine-tuning  
1. **Modify these variables in `train-bert.py`:**  
```python  
checkpoint = "bert-base-cased"  # Pre-trained model  
data_path = "dataset/dataset-all.csv"  # Dataset path  
num_epochs = 10  # Training epochs  
batch_size = 32  # Batch size  

Run training:
python train-bert.py  

DistilBERT Training
Edit train-distilbert.py similarly:
python
Copy
checkpoint = "distilbert-base-cased"  # Pre-trained model  
data_path = "dataset/dataset-all.csv"  # Dataset path  
num_epochs = 10  # Training epochs  
batch_size = 32  # Batch size  

Start training:
python train-distilbert.py  

📂 Datasets
dataset-all.csv (3,790 entries):
text,labels  
"Question: Does product X have... Answer:...",0  
"Question: Is item A... Answer:...",1  

dataset-match-score.csv (1,417 entries):
text,labels  
"Question: Does smartphone... Match Score:...",0  
"Question: Can device Y... Match Score:...",1  


generating-impossible-queries/
├── database/
│   ├── connect.py           # MySQL connection handler
│   └── dump.sql             # Database schema + sample data
├── docs/
│   └── images/              # Architecture diagrams
├── examples/
│   └── pipeline_example.ipynb  # Full workflow demo
├── src/
│   ├── models/              # AI model implementations
│   ├── preprocessing/       # Data processing modules
│   ├── train-bert.py        # BERT fine-tuning script
│   └── full_pipeline.py     # Full implementation example
├── .env.example             # Environment template
└── requirements.txt         # Python dependencies
