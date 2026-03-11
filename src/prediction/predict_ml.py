import pickle

from src.preprocessing.clean_text import clean_text

model = pickle.load(open("models/ml_model.pkl","rb"))

vectorizer = pickle.load(open("models/vectorizer.pkl","rb"))

def predict(text):

    text = clean_text(text)

    vec = vectorizer.transform([text])

    pred = model.predict(vec)[0]

    return pred