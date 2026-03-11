import os
import pickle

from src.preprocessing.clean_text import clean_text

MODEL_PATH = "models/ml_model.pkl"
VEC_PATH = "models/vectorizer.pkl"

# If model not exist -> train automatically
if not os.path.exists(MODEL_PATH):

    from src.training.train_ml import train_model
    train_model()

with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)

with open(VEC_PATH, "rb") as f:
    vectorizer = pickle.load(f)


def predict(text):

    text = clean_text(text)

    vec = vectorizer.transform([text])

    prediction = model.predict(vec)[0]

    return prediction