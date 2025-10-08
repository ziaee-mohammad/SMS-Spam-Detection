📱 SMS Spam Detection

🧠 Project Overview

This project focuses on classifying SMS messages as Spam or Ham (Non-Spam) using Natural Language Processing (NLP) and Machine Learning techniques. It involves text cleaning, visualization, feature engineering, and model comparison to identify the most accurate spam detection approach.

---

📊 Dataset

Source: SMS Spam Collection dataset

Samples: 5,572 messages

Labels:

ham → legitimate message

spam → unwanted or promotional message



---

🧹 Data Preprocessing

Converted text to lowercase

Removed punctuation, special characters, and stopwords

Applied stemming using PorterStemmer

Tokenized and vectorized using TF-IDF


---

📈 Exploratory Data Analysis (EDA)

Checked data balance between spam and ham messages

Visualized message lengths and word counts

Generated WordClouds for spam and ham messages

Analyzed correlations between numeric features


---

🧩 Feature Engineering

Extracted features such as:

num_characters

num_words

num_sentences


Converted text data into numerical form using TF-IDF Vectorizer


---

🤖 Model Building & Evaluation

Tested multiple models including:

Multinomial Naive Bayes ✅ (Best Accuracy ≈ 98%)

Gaussian Naive Bayes

Support Vector Machine (SVM)


Evaluation Metrics:

Accuracy

Precision

Recall

F1-Score


---

💾 Deployment

The trained model (model.pkl) and vectorizer (vectorizer.pkl) are saved for integration into a web app (app.py).


---

📦 Requirements

pandas  
numpy  
scikit-learn  
nltk  
matplotlib  
wordcloud  
pickle


---

✨ Results

Achieved ~98% accuracy using Multinomial Naive Bayes with TF-IDF features.

Demonstrated reliable spam detection across diverse text samples.


---

👨‍💻 Author
Mohammad Ziaee
Moha2012zia@gmail.com



---

می‌خوای نسخه‌ی «مختصرتر» هم برات بسازم که فقط تو GitHub خلاصه‌ی چشم‌گیر بالا نشون بده (برای موبایل و جستجو خیلی مؤثره)؟
