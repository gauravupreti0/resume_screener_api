import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
import joblib

df = pd.read_csv("data/sample_resumes.csv")
X = df['text']
Y = df['label']

vectorizer = TfidfVectorizer()
X_vec = vectorizer.fit_transform(X)

clf = RandomForestClassifier()
clf.fit(X_vec, Y)

joblib.dump(clf, 'model/resume_classifier.pkl')
joblib.dump(vectorizer, 'model/resume_vectorizer.pkl')

print("✅ Model training complete. Files saved in /model")