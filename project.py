import pandas as pd
import re
import nltk

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix

nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

ps = PorterStemmer()
stop_words = set(stopwords.words('english'))

def clean_text(text):
    text = re.sub(r'[^a-zA-Z]', ' ', text)
    text = text.lower()
    words = text.split()
    words = [ps.stem(word) for word in words if word not in stop_words]
    return ' '.join(words)

fake = pd.read_csv("Fake.csv")
true = pd.read_csv("True.csv")

fake['label'] = 0
true['label'] = 1

data = pd.concat([fake, true], axis=0).reset_index(drop=True)
data = data[['title', 'text', 'label']]
data['combined'] = data['title'] + " " + data['text']
data['cleaned_text'] = data['combined'].apply(clean_text)

vectorizer = TfidfVectorizer(max_df=0.7, min_df=2, ngram_range=(1,2))
X = vectorizer.fit_transform(data['cleaned_text'])
y = data['label']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))

def check_news(text):
    cleaned = clean_text(text)
    vector = vectorizer.transform([cleaned])
    return "Real" if model.predict(vector)[0] == 1 else "Fake"

print(check_news("India launches a new space mission next year."))#real
print(check_news("NASA found aliens hiding in the moon base."))#fake
print(check_news("The Prime Minister announced a new economic policy today."))  # Real
print(check_news("Scientists discovered a cure for cancer in just one week."))   # Likely Fake
print(check_news("Facebook plans to launch a new feature for reels and stories."))  # Real
print(check_news("Aliens have landed in Russia and taken over the Kremlin."))      # Fake
print(check_news("COVID-19 vaccination for all adults starts from next Monday."))  # Real
print(check_news("Donald Trump secretly owns Google and Amazon shares."))          # Fake
print(check_news("World Bank approves $500M loan for Indian infrastructure."))     # Real
print(check_news("NASA confirms two moons orbiting Earth after new discovery."))   # Fake
print(check_news("Supreme Court passes new data privacy regulation bill."))        # Real
print(check_news("Drinking cow urine daily prevents all kinds of diseases."))      # Likely Fake\
print(check_news("Hinduism is not what people are practicing it today")) #BHOT JYDA REAL