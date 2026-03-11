import sys
import os

import pandas as pd
import pickle


from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from src.preprocessing.clean_text import clean_text
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
df = pd.read_csv("data/train.csv")

df["text"] = df["Title"] + " " + df["Description"]

df["text"] = df["text"].apply(clean_text)

X = df["text"]

y = df["Class Index"]

vectorizer = TfidfVectorizer(max_features=5000)

X_vec = vectorizer.fit_transform(X)

model = LogisticRegression()

model.fit(X_vec,y)

pickle.dump(model,open("models/ml_model.pkl","wb"))

pickle.dump(vectorizer,open("models/vectorizer.pkl","wb"))

print("Training completed")