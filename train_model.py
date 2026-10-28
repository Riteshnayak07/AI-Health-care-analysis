"""
PulseAI - ML Model Training Pipeline
Trains TF-IDF + Logistic Regression Classifier on multi-class clinical symptom patterns.
Saves model artifacts for real-time inference in Flask.
"""
import os, pickle
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

BASE = os.path.dirname(os.path.abspath(__file__))
dataset_path = os.path.join(BASE, "dataset", "symptoms_disease.csv")
data = pd.read_csv(dataset_path)

print(f"Loaded {len(data)} training samples across {data['disease'].nunique()} conditions.")

X = data["symptoms"]
y = data["disease"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

pipeline = Pipeline([
    ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=1, sublinear_tf=True)),
    ("clf", LogisticRegression(max_iter=3000, C=1.5, class_weight="balanced"))
])

pipeline.fit(X_train, y_train)
y_pred = pipeline.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f"\nModel Validation Accuracy: {acc * 100:.2f}%\n")
print(classification_report(y_test, y_pred, zero_division=0))

# Save to both standard locations for backward compatibility
os.makedirs(os.path.join(BASE, "model"), exist_ok=True)
os.makedirs(os.path.join(BASE, "models"), exist_ok=True)

with open(os.path.join(BASE, "model", "healthcare_model.pkl"), "wb") as f:
    pickle.dump(pipeline, f)

# Also save standalone vectorizer & model for nlp module
tfidf_step = pipeline.named_steps["tfidf"]
clf_step = pipeline.named_steps["clf"]
with open(os.path.join(BASE, "models", "tfidf_vectorizer.pkl"), "wb") as f:
    pickle.dump(tfidf_step, f)
with open(os.path.join(BASE, "models", "symptom_model.pkl"), "wb") as f:
    pickle.dump(clf_step, f)

print("Saved model artifacts to model/healthcare_model.pkl and models/")
