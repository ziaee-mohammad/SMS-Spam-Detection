# 📩 SMS Spam Detection

An end‑to‑end **NLP** project for classifying SMS messages as **spam** or **ham**.  
Includes text cleaning, TF‑IDF feature extraction, classical ML baselines (Naive Bayes, Logistic Regression, Linear SVM), optional deep models, and rigorous evaluation with Accuracy, Precision, Recall, F1‑score, and ROC/PR curves.

---

## 📖 Overview
This repository implements a clean and reproducible pipeline for SMS spam detection:
- Normalization & cleaning (URLs, mentions, punctuation, numbers)
- Tokenization, stopword removal, **lemmatization**
- Feature extraction with **TF‑IDF** (n‑grams)
- Model training & hyperparameter tuning
- Evaluation and error analysis (confusion matrix, top features)

---

## 🗂️ Dataset
- Typical CSV layout under `Dataset/`:
```
Dataset/
├─ train.csv
└─ test.csv
```
- **Columns** (example): `id`, `text`, `label` (`spam`/`ham` or `1/0`).  
- If you use a public dataset (e.g., UCI SMS Spam Collection), cite its source in this README.

> Check **class balance**; use stratified splits and class weights if needed.

---

## 🧹 Preprocessing
- Lowercasing and whitespace normalization  
- Remove URLs, emails, mentions, and non‑alphanumeric chars (configurable)  
- Tokenization, stopword removal (NLTK)  
- Lemmatization (WordNet) or stemming  
- **TF‑IDF** with n‑grams (1–2 or 1–3), `min_df` filtering, `max_features` cap

---

## 🧠 Models
- **Multinomial Naive Bayes** — fast and effective for sparse text
- **Logistic Regression** — strong linear baseline with regularization
- **Linear SVM** — max‑margin classifier; often top performer on TF‑IDF
- *(Optional)* Light models with char‑ngrams; deep models if you extend

---

## 📈 Evaluation (replace with your numbers)
| Model | Accuracy | Precision | Recall | F1 |
|------|---------:|----------:|------:|---:|
| Naive Bayes | 0.97 | 0.96 | 0.97 | 0.96 |
| Logistic Regression | 0.98 | 0.98 | 0.98 | 0.98 |
| Linear SVM | 0.98 | 0.98 | 0.98 | 0.98 |

Export:
- Confusion matrix, ROC/PR curves
- Per‑class metrics and support
- Top indicative features (for linear models)

---

## 🧩 Repository Structure (suggested)
```
SMS-Spam-Detection/
├─ Dataset/                      # train/test CSVs
├─ Notebook/
│  └─ SMS_Spam_Detection.ipynb   # main notebook
├─ src/                          # optional scripts
│  ├─ data.py                    # loading/cleaning
│  ├─ features.py                # TF-IDF / vectorizers
│  ├─ train.py                   # training & tuning
│  └─ eval.py                    # metrics & plots
├─ reports/figures/              # CM, ROC/PR, feature plots
├─ requirements.txt
├─ .gitignore
└─ README.md
```

---

## ⚙️ Setup & Usage
1) **Clone & install**
```bash
git clone https://github.com/ziaee-mohammad/SMS-Spam-Detection.git
cd SMS-Spam-Detection
pip install -r requirements.txt
```

2) **Run notebook**
```bash
jupyter notebook Notebook/SMS_Spam_Detection.ipynb
```

3) **(Optional) Run scripts**
```bash
python -m src.train --model "svm" --ngrams 1,2 --max_features 200000
python -m src.eval  --report
```

---

## 📦 Requirements (example)
```
pandas
numpy
scikit-learn
nltk
matplotlib
seaborn
```
> If you need lemmatization: `python -m nltk.downloader stopwords wordnet punkt`

---

## ✅ Good Practices
- Use **stratified** train/test split (preserve spam ratio)  
- Keep vectorizer + model in a single `Pipeline` to avoid leakage  
- Fix random seeds for reproducibility  
- Save vectorizer/model artifacts for deployment

---

## 🏷 Tags
```
data-science
machine-learning
nlp
spam-detection
text-mining
classification
python
scikit-learn
tf-idf
``

---

## 👤 Author
**Mohammad Ziaee** — Computer Engineer | AI & Data Science  
📧 moha2012zia@gmail.com  
🔗 https://github.com/ziaee-mohammad
👉 Instagram: [@ziaee_mohammad](https://www.instagram.com/ziaee_mohammad/)

---

## 📜 License
MIT — free to use and adapt with attribution.

