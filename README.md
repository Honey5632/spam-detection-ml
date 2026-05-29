# 📧 SMS Spam Detection — Machine Learning Project

> Classifying SMS messages as **Spam or Ham** using TF-IDF + Naive Bayes with **90–98% accuracy**

---

## 📌 Overview

This project builds a complete **text classification pipeline** to detect spam SMS messages. It uses TF-IDF vectorization with bigrams to extract features from raw text, and a Multinomial Naive Bayes classifier to predict whether a message is spam or legitimate (ham).

The project includes two scripts:
- `main.py` — trains the model and saves it to disk
- `predict_spam.py` — loads the saved model and runs live predictions in the terminal

---

## 📊 Dataset

| Property | Value |
|---|---|
| File | `spam.csv` |
| Encoding | latin-1 |
| Columns used | `v1` (label), `v2` (text) |
| Labels | `ham` → 0, `spam` → 1 |
| Split | 80% train / 20% test (stratified) |

> 📥 **Dataset source:** [SMS Spam Collection — Kaggle](https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset)
> Download and place `spam.csv` in the project root before running.

---

## 🛠️ Tech Stack

![Python](https://img.shields.io/badge/Python-3.x-blue?logo=python)
![scikit-learn](https://img.shields.io/badge/scikit--learn-latest-orange?logo=scikit-learn)
![pandas](https://img.shields.io/badge/pandas-latest-150458?logo=pandas)

```
pandas · scikit-learn · joblib
```

---

## ⚙️ How It Works

```
Raw SMS Text
     │
     ▼
Load & Clean CSV
  └── Keep label + text columns
  └── Drop nulls
  └── Encode: ham=0, spam=1
     │
     ▼
Train / Test Split (80/20, stratified)
     │
     ▼
TF-IDF Vectorization
  └── lowercase=True
  └── stop_words='english'
  └── max_features=5000
  └── ngram_range=(1, 2)  ← unigrams + bigrams
     │
     ▼
Multinomial Naive Bayes
  └── model.fit(X_train_tfidf, y_train)
     │
     ▼
Evaluate: Accuracy · Precision · Recall · F1
     │
     ▼
Save model + vectorizer via joblib
  └── spam_model.pkl
  └── vectorizer.pkl
```

---

## 📈 Results

| Metric | Score |
|---|---|
| **Accuracy** | **90% – 98%** |
| Precision (spam) | High |
| Recall (spam) | High |
| F1-score (spam) | High |

> Exact scores vary slightly by run. The stratified split ensures consistent class distribution between train and test sets.

---

## 📁 Project Structure

```
spam-detection-ml/
│
├── main.py              # Train model + save to disk
├── predict_spam.py      # Load model + live terminal predictions
├── README.md            # Project documentation
│
├── spam.csv             # Dataset (download from Kaggle — not included)
│
└── model/
    ├── spam_model.pkl   # Saved Naive Bayes model
    └── vectorizer.pkl   # Saved TF-IDF vectorizer
```

---

## 🚀 How to Run

**1. Clone the repository**
```bash
git clone https://github.com/Honey5632/spam-detection-ml.git
cd spam-detection-ml
```

**2. Install dependencies**
```bash
pip install pandas scikit-learn joblib
```

**3. Add the dataset**
Download `spam.csv` from [Kaggle](https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset) and place it in the project root.

**4. Train the model**
```bash
python main.py
```
This will print accuracy + classification report and save `spam_model.pkl` and `vectorizer.pkl`.

**5. Run live predictions**
```bash
python predict_spam.py
```

---

## 💬 Live Prediction Demo

```
Enter message (or type 'exit' to quit): Congratulations! You've won a free iPhone. Click here to claim now.
Prediction: SPAM
Spam probability: 97.43 %

Enter message (or type 'exit' to quit): Hey, are we still meeting at 6pm?
Prediction: NOT SPAM
Spam probability: 0.82 %

Enter message (or type 'exit' to quit): exit
```

---

## 💡 Why TF-IDF + Naive Bayes?

| Choice | Reason |
|---|---|
| **TF-IDF** | Converts raw text to weighted numerical features — rare spam keywords like "FREE", "WIN", "CLAIM" get high scores |
| **Bigrams `(1,2)`** | Captures two-word phrases like "click here", "free prize", "you won" that single words miss |
| **Multinomial NB** | Works exceptionally well on text data, fast to train, and performs great even on small datasets |
| **Stratified split** | Ensures spam/ham ratio is preserved in both train and test sets — important since spam is a minority class |

---

## 🔧 Possible Improvements

- [ ] Try `LogisticRegression` or `SVM` for comparison
- [ ] Add a Flask/FastAPI web interface
- [ ] Deploy on Streamlit for a live demo
- [ ] Experiment with `max_features` and `ngram_range` tuning
- [ ] Add confusion matrix visualization

---

## 🙋 Author

**Honey**
[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-blue?logo=linkedin)](https://www.linkedin.com/in/honey-rana-6748b938a)
[![GitHub](https://img.shields.io/badge/GitHub-Follow-black?logo=github)](https://github.com/Honey5632)

---

## 📄 License

This project is open source and available under the [MIT License](LICENSE).
