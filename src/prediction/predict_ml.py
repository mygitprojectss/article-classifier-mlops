import pickle

with open("models/ml_model.pkl","rb") as f:
    model = pickle.load(f)

with open("models/vectorizer.pkl","rb") as f:
    vectorizer = pickle.load(f)

def predict(text):

    text_vec = vectorizer.transform([text])

    prediction = model.predict(text_vec)[0]

    return prediction