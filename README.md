# 📰 Fake News Detection using Machine Learning

A machine learning project to detect fake and real news articles using **TF-IDF vectorization** and **Logistic Regression**.

---

## 📂 Dataset Used
- [`Fake.csv`](https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset)
- [`True.csv`](https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset)

---

## ⚙️ Technologies
- Python
- scikit-learn
- pandas
- nltk
- TF-IDF
- Logistic Regression / Passive Aggressive Classifier

---

## 🧠 How it Works
1. Combines `title` and `text` fields
2. Cleans the text (stopword removal + stemming)
3. Converts to TF-IDF vectors
4. Trains a model on real vs fake labels
5. Predicts new custom input using `check_news()` function

---

## 📈 Accuracy
Achieved ~98.6% test accuracy  
Confusion Matrix shows strong balance between real and fake predictions.

---

## ▶️ Run it Locally

```bash
git clone https://github.com/yourusername/fake-news-detector.git
cd fake-news-detector
pip install -r requirements.txt
python fake_news_detector.py
