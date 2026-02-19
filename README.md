# 📧 Spam Mail Prediction Tool

A complete Machine Learning project that detects whether a message/email is **Spam** or **Ham (Not Spam)** using Natural Language Processing (NLP) techniques and supervised learning algorithms.

---

## 🚀 Project Overview

Spam emails are one of the most common cybersecurity and communication issues today.  
This project builds an intelligent classification system that automatically identifies spam messages based on textual patterns.

The model processes raw text data, converts it into numerical features, trains a machine learning classifier, and predicts whether a message is spam or not.

This notebook demonstrates the **complete ML pipeline**:
- Data Cleaning
- Text Preprocessing
- Feature Engineering
- Model Training
- Model Evaluation
- Prediction on New Data

---

## 🧠 Problem Statement

Given a text message or email:

👉 Determine whether it is:
- **Spam (Unwanted/Promotional/Fraudulent Message)**
- **Ham (Legitimate Message)**

This is a **Binary Classification Problem**.

---

## 📊 Dataset Information

The dataset contains:
- `Label` → Spam / Ham
- `Message` → Text content

Typical examples:

| Message | Label |
|----------|--------|
| "Congratulations! You won a free ticket!" | Spam |
| "Let's meet tomorrow at 5 PM." | Not Spam |

---

## 🔬 Machine Learning Workflow

### 1️⃣ Data Preprocessing

- Removed null values
- Converted text to lowercase
- Removed punctuation and special characters
- Tokenization
- Stopword removal
- Optional: Stemming / Lemmatization

Why?  
Because machine learning models cannot directly understand raw text.

---

### 2️⃣ Feature Engineering

Text data was converted into numerical format using:

- **Bag of Words (CountVectorizer)**
- or **TF-IDF Vectorizer**

This transforms text into a structured matrix suitable for ML algorithms.

---

### 3️⃣ Train-Test Split

Dataset split into:
- 80% Training Data
- 20% Testing Data

Using:
```python
train_test_split()
```
---

## 4️⃣ Model Training

The following classifier was used:

- ✅ **Naive Bayes** (commonly used for text classification)  
  **or**
- **Logistic Regression** (if used in the notebook)

The model was trained on **vectorized text features** generated using Bag of Words or TF-IDF.

---

## 5️⃣ Model Evaluation

The model performance was evaluated using the following metrics:

- 📌 **Accuracy Score**
- 📌 **Confusion Matrix**
- 📌 **Precision**
- 📌 **Recall**
- 📌 **F1 Score**

### 📊 Example:
```
Accuracy: 97%
```

### 🔎 Understanding Confusion Matrix

The Confusion Matrix helps analyze:

- ✅ **True Positives (TP)** → Spam correctly predicted as Spam  
- ✅ **True Negatives (TN)** → Ham correctly predicted as Ham  
- ❌ **False Positives (FP)** → Ham incorrectly predicted as Spam  
- ❌ **False Negatives (FN)** → Spam incorrectly predicted as Ham  

---

## 📈 Results

The model achieved **high accuracy** in detecting spam messages.

### 🚀 Strengths

- ⚡ Fast prediction
- 🪶 Lightweight model
- 🎯 Good generalization on unseen messages

---

## 🛠 Technologies & Libraries Used

- Python 3.x
- NumPy
- Pandas
- Matplotlib
- Seaborn (if used)
- Scikit-learn
- Jupyter Notebook

---

## 📂 Project Structure
```
📁 Spam-Mail-Prediction
│
├── Spam Mail Prediction Tool.ipynb
├── README.md
└── dataset.csv (if included)
```

---

## ▶️ How to Run This Project

### Step 1: Clone Repository

```bash
git clone https://github.com/your-username/spam-mail-prediction.git
```

### Step 2: Navigate to Project Folder
```
cd spam-mail-prediction
```
### Step 3: Install Dependencies
```
pip install -r requirements.txt
```

Or manually install:
```
pip install pandas numpy scikit-learn matplotlib
```
### Step 4: Run Jupyter Notebook
```
jupyter notebook
```

Open:
```
Spam Mail Prediction Tool.ipynb
```

Run all cells.

## 🧪 Example Prediction
📥 Input
```
"Congratulations! You have been selected for a free gift voucher."
```
📤 Output
```
Spam
```
📥 Input
```
"Are we still meeting today?"
```
📤 Output
```
Not Spam
```
