import streamlit as st
import pickle
import re

# Load trained model
with open("sentiment_model.pkl", "rb") as file:
    model = pickle.load(file)

with open("vectorizer.pkl", "rb") as file:
    vectorizer = pickle.load(file)

# Text cleaning
def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    return text

# Page configuration
st.set_page_config(
    page_title="AI Sentiment Analyzer",
    page_icon="🤖",
    layout="wide"
)

# Styling
st.markdown("""
<style>
.stApp {
    background-color: #101827;
}
h1 {
    color: #38bdf8;
}
</style>
""", unsafe_allow_html=True)

# Application heading
st.title("🤖 AI Text Sentiment Analyzer")

st.write(
    "Analyze emotions and opinions from text "
    "using Machine Learning and NLP."
)

st.divider()

# User input
user_text = st.text_area(
    "Enter your text:",
    placeholder="Example: I really love this product!"
)

if st.button("Analyze Sentiment"):

    if user_text.strip():

        cleaned = clean_text(user_text)

        transformed = vectorizer.transform([cleaned])

        prediction = model.predict(transformed)[0]

        probabilities = model.predict_proba(transformed)[0]

        st.subheader("Sentiment Result")

        if prediction.lower() == "positive":
            st.success("😊 Positive Sentiment")

        elif prediction.lower() == "negative":
            st.error("😞 Negative Sentiment")

        else:
            st.info("😐 Neutral Sentiment")

        st.metric(
            "Confidence Score",
            f"{max(probabilities) * 100:.2f}%"
        )

        st.subheader("Probability Analysis")

        for label, probability in zip(
            model.classes_, probabilities
        ):
            st.write(f"{label}: {probability * 100:.2f}%")
            st.progress(float(probability))

    else:
        st.warning("Please enter some text!")
