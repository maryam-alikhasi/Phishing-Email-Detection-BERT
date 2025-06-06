# Phishing-Email-Detection-BERT

This project implements a phishing email detection system using **BERT** (Bidirectional Encoder Representations from Transformers). The goal is to train a fine-tuned transformer-based model that can distinguish between phishing and legitimate emails based on their content.

---

## Objective

Train a fine-tuned BERT-based classifier to identify phishing emails from regular emails using a labeled dataset.

---

## Dataset

- **Name**: Phishing Email Dataset  
- **Source**: [Kaggle Dataset - naserabdullahalam/phishing-email-dataset](https://www.kaggle.com/datasets/naserabdullahalam/phishing-email-dataset)  
- **Columns**:
  - `body`: Email content
  - `label`: Target label (`phishing` or `legitimate`)

---

## Model

- Pretrained transformer: `bert-base-uncased`
- Fine-tuned for **binary text classification** (phishing vs normal)
---

## Architecture & Workflow

1. **Data Preprocessing**
   - Load CSV
   - Tokenize email bodies using `BertTokenizer`
   - Encode labels with `LabelEncoder`
   - Train/test split

2. **Custom Dataset Class**
   - PyTorch `Dataset` wrapper for tokenized data

3. **Model Setup**
   - `BertForSequenceClassification` with 2 output labels
   - Optimizer: `AdamW`
   - Scheduler: `get_linear_schedule_with_warmup`
   - Device: CPU / GPU (automatically detected)

4. **Training**
   - Train BERT for 5 epochs
   - Loss: `CrossEntropyLoss`

5. **Evaluation**
   - Metrics: `Accuracy`, `Precision`, `Recall`, `F1-Score` using `sklearn`

6. **Prediction**
   - Inference function `predict_text(text)` for new unseen email content

---


## Requirements

```bash
transformers
torch
scikit-learn
pandas
tqdm
````

## How to Run

### 1. Download Dataset

In Google Colab:

```python
!pip install kaggle
from google.colab import files
files.upload()  # Upload kaggle.json
!mkdir -p ~/.kaggle
!cp kaggle.json ~/.kaggle/
!chmod 600 ~/.kaggle/kaggle.json

!kaggle datasets download -d naserabdullahalam/phishing-email-dataset
!unzip phishing-email-dataset.zip
```

### 2. Run Notebook

Open and execute `Phishing-BERT.ipynb`.

