import streamlit as st
import tensorflow as tf

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Next Word Predictor",
    page_icon="✨",
    layout="centered"
)

# ---------------- CUSTOM CSS ----------------
st.markdown("""
<style>

body {
    background-color: #f8f9fb;
}

/* Input box styling */
.stTextInput input {
    background-color: #dfe3e8 !important;
    color: black !important;
    border-radius: 12px !important;
    padding: 12px !important;
    border: 1px solid #c9ced6 !important;
}

/* Button styling */
.stButton>button {
    background-color: #a8b2c1;
    color: white;
    border-radius: 10px;
    border: none;
    padding: 0.5rem 1rem;
    font-weight: bold;
}

.stButton>button:hover {
    background-color: #8e99aa;
}

</style>
""", unsafe_allow_html=True)

# ---------------- SIDEBAR ----------------
with st.sidebar:
    st.header("📘 About")

    st.write("""
    ### Next Word Prediction App

    This application predicts the next possible word
    based on the input sentence using a trained
    TensorFlow deep learning model.

    ### ⚙️ Tech Stack
    - Python
    - TensorFlow / Keras
    - NLP
    - Streamlit

    ### 🧠 Model Used
    LSTM-based Sequential Neural Network trained on text data.
    """)

# ---------------- MAIN UI ----------------
st.title("✨ Next Word Prediction")

st.write("Enter a sentence and predict the next possible word.")

user_input = st.text_input(
    "Type your sentence here"
)

if st.button("Predict"):

    # Replace this with your actual prediction function
    prediction = "example_word"

    st.success(f"Predicted Next Word: **{prediction}**")
