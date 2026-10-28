import pickle
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

BASE = Path(__file__).resolve().parents[1]
MODEL_DIR = BASE / "models"
MODEL_DIR.mkdir(exist_ok=True)

texts = [
    "fever cough sore throat body ache",
    "high fever headache tiredness body pain",
    "cough runny nose fever fatigue",
    "sore throat cough mild fever",
    "headache nausea vomiting light sensitivity",
    "severe headache nausea dizziness",
    "head pain nausea and vomiting",
    "headache dizziness and sensitivity",
    "stomach pain diarrhea vomiting",
    "abdominal pain loose stools",
    "diarrhea stomach cramps nausea",
    "vomiting abdominal pain diarrhea",
    "itchy skin rash red patches",
    "skin rash itching redness",
    "red itchy rash on skin",
    "skin irritation and itching",
    "chest pain shortness of breath",
    "difficulty breathing chest discomfort",
    "breathing problem chest pain",
    "shortness of breath and chest discomfort",
]

labels = (
    ["Flu-like symptom pattern"] * 4
    + ["Headache/migraine-like pattern"] * 4
    + ["Gastrointestinal symptom pattern"] * 4
    + ["Skin-related symptom pattern"] * 4
    + ["Respiratory/chest symptom pattern"] * 4
)

vectorizer = TfidfVectorizer(ngram_range=(1, 2), lowercase=True)
x = vectorizer.fit_transform(texts)

model = LogisticRegression(max_iter=2000)
model.fit(x, labels)

with open(MODEL_DIR / "tfidf_vectorizer.pkl", "wb") as f:
    pickle.dump(vectorizer, f)
with open(MODEL_DIR / "symptom_model.pkl", "wb") as f:
    pickle.dump(model, f)

print("NLP model trained successfully.")
