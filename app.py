import streamlit as st
import joblib

# Load model and tfidf
model = joblib.load("model.pkl")
tfidf = joblib.load("tfidf.pkl")


# Text cleaning function (IMPORTANT: same as training preprocessing)
import re

def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"<.*?>", "", text)
    text = re.sub(r"\S+@\S+", "", text)
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    text = re.sub(r"\s+", " ", text)
    return text


# UI Title
st.title("🚨 Fake Job Detection System")

st.write("Enter a job description to check if it's REAL or FAKE")

# Input box
user_input = st.text_area("Job Description")

# Predict button
if st.button("Predict"):

    cleaned = clean_text(user_input)

    vector = tfidf.transform([cleaned])

    prediction = model.predict(vector)[0]

    # Output
    if prediction == 1:
        st.error("⚠️ FAKE JOB DETECTED")
    else:
        st.success("✅ GENUINE JOB")
