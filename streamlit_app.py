import streamlit as st

from src.prediction.predict_ml import predict
from src.database.db import create_db, save_prediction

create_db()

st.title("📰 News Article Classifier")

st.write("Classify news articles using ML models")

text = st.text_area("Enter News Article")

model_type = st.selectbox(
    "Choose Model",
    ["Machine Learning"]
)

if st.button("Predict"):

    prediction = predict(text)

    labels = {
        1: "World",
        2: "Sports",
        3: "Business",
        4: "Sci/Tech"
    }

    result = labels.get(prediction, "Unknown")

    save_prediction(text, result, model_type)

    st.success(f"Prediction: {result}")