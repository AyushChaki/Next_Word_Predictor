import streamlit as st
import pickle
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# ------------------------------
# Page Config
# ------------------------------
st.set_page_config(
    page_title="Next Word Predictor",
    page_icon="✍️",
    layout="centered"
)

# ------------------------------
# Custom CSS
# ------------------------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Merriweather:wght@300;400;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Merriweather', serif;
}

.stApp {
    background: linear-gradient(to bottom right, #f7f3ef, #e8ecef);
    color: #1e1e1e;
}

.main-title {
    text-align: center;
    font-size: 3rem;
    font-weight: 700;
    margin-top: 20px;
    color: #2c2c2c;
}

.sub-text {
    text-align: center;
    font-size: 1rem;
    color: #5c5c5c;
    margin-bottom: 30px;
}

.prediction-box {
    background-color: rgba(255,255,255,0.8);
    padding: 20px;
    border-radius: 15px;
    text-align: center;
    margin-top: 20px;
    box-shadow: 0px 4px 10px rgba(0,0,0,0.1);
}

.stButton > button {
    width: 100%;
    border-radius: 10px;
    background-color: #cfd8dc;
    color: black;
    font-weight: bold;
    border: none;
    padding: 10px;
}

.stButton > button:hover {
    background-color: #b0bec5;
}

.stTextInput > div > div > input {
    border-radius: 10px;
    padding: 10px;
}
</style>
""", unsafe_allow_html=True)

# ------------------------------
# Load Model
# ------------------------------
model = load_model("lstm_model.h5")

with open("tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)

with open("max_len.pkl", "rb") as f:
    max_len = pickle.load(f)

# ------------------------------
# Header
# ------------------------------
st.markdown(
    '<div class="main-title">Next Word Predictor</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="sub-text">LSTM powered intelligent text prediction</div>',
    unsafe_allow_html=True
)

# ------------------------------
# Prediction Function
# ------------------------------
def predict_next_word(text):

    token_list = tokenizer.texts_to_sequences([text])[0]

    token_list = pad_sequences(
        [token_list],
        maxlen=max_len - 1,
        padding='pre'
    )

    predicted = model.predict(token_list, verbose=0)

    predicted_word_index = np.argmax(predicted)

    output_word = ""

    for word, index in tokenizer.word_index.items():
        if index == predicted_word_index:
            output_word = word
            break

    return output_word

# ------------------------------
# User Input
# ------------------------------
input_text = st.text_input(
    "Enter your text",
    placeholder="Type your sentence here..."
)

# ------------------------------
# Prediction Button
# ------------------------------
if st.button("Predict Next Word"):

    if input_text.strip() == "":
        st.warning("Please enter some text.")

    else:

        next_word = predict_next_word(input_text)

        st.markdown(
            f"""
            <div class="prediction-box">
                <h3>Predicted Next Word</h3>
                <h2>{next_word}</h2>
            </div>
            """,
            unsafe_allow_html=True
        )

# ------------------------------
# Footer
# ------------------------------
st.markdown(
    """
    <br><br>
    <div style='text-align:center; color:gray; font-size:14px;'>
        Built with Streamlit & TensorFlow
    </div>
    """,
    unsafe_allow_html=True
)
