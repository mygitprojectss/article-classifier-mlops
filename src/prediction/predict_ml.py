import os
import pickle

from src.preprocessing.clean_text import clean_text


# auto train if model not present
if not os.path.exists("models/ml_model.pkl"):

    from src.training.train_ml import train_model
    train_model()


with open("models/ml_model.pkl","rb") as f:
    model = pickle.load(f)

with open("models/vectorizer.pkl","rb") as f:
    vectorizer = pickle.load(f)


def predict(text):

    text = clean_text(text)

    vec = vectorizer.transform([text])

    prediction = model.predict(vec)[0]

    return prediction