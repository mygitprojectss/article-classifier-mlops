import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import pandas as pd
import pickle

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from src.preprocessing.clean_text import clean_text


def train_model():

    # load dataset
    df = pd.read_csv("data/train.csv")

    # combine title + description
    df["text"] = df["Title"] + " " + df["Description"]

    # clean text
    df["text"] = df["text"].apply(clean_text)

    X = df["text"]
    y = df["Class Index"]

    # vectorization
    vectorizer = TfidfVectorizer(max_features=5000)
    X_vec = vectorizer.fit_transform(X)

    # model
    model = LogisticRegression(max_iter=200)

    model.fit(X_vec, y)

    # create models folder
    os.makedirs("models", exist_ok=True)

    # save model
    with open("models/ml_model.pkl", "wb") as f:
        pickle.dump(model, f)

    with open("models/vectorizer.pkl", "wb") as f:
        pickle.dump(vectorizer, f)

    print("Training completed and model saved")


if __name__ == "__main__":
    train_model()