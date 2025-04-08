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

bash
Copy
# Clone repository
git clone https://github.com/AymaneHassini/generating-impossible-queries.git
cd generating-impossible-queries

# Install dependencies
pip install -r requirements.txt

# Configure environment
cp .env.example .env  # Update with your API keys and DB credentials
🗃️ Database Setup

bash
Copy
# Create MySQL database
mysql -u root -p -e "CREATE DATABASE prods"

# Import schema and data
mysql -u root -p prods < database/dump.sql

# Verify installation
mysql -u root -p -e "USE prods; SHOW TABLES;"

💡 Usage Example
![Full implementation example:](pipeline_example.ipynb)

📂 Project Structure
generating-impossible-queries/
├── database/
│   ├── connect.py           # MySQL connection handler
│   └── dump-example-database.sql             # Database schema + sample data
├── docs/
│   └── images/              # Architecture diagrams
├── examples/
│   └── pipeline_example.ipynb  # Full workflow demo
├── src/
│   ├── models/              # AI model implementations
│   ├── preprocessing/       # Data processing modules
│   ├── train-bert.py        # BERT fine-tuning script
│   └── full_pipeline.py     # Main query interface
├── .env.example             # Environment template
└── requirements.txt         # Python dependencies