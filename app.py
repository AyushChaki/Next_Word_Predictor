import streamlit as st
import tensorflow as tf
import numpy as np
import pickle
import os
from tensorflow.keras.preprocessing.sequence import pad_sequences


# ================= PAGE CONFIG =================
st.set_page_config(
    page_title="Next Word Prediction",
    page_icon="🪶",
    layout="wide"
)


# ================= CUSTOM CSS =================
st.markdown("""
<style>

/* Main background */
.stApp {
    background-color: #f7f5f0;
    color: #111111;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background-color: #eeeae2;
    border-right: 1px solid #cfc8bc;
}

section[data-testid="stSidebar"] * {
    color: #111111 !important;
}

/* Header transparent */
header[data-testid="stHeader"] {
    background: transparent;
}

/* Main container */
.block-container {
    padding-top: 5rem;
    padding-left: 4rem;
    padding-right: 4rem;
}

/* Title */
.main-title {
    font-family: Georgia, serif;
    font-size: 64px;
    font-weight: 800;
    color: #111111;
    margin-bottom: 10px;
}

/* Subtitle */
.subtitle {
    font-family: Georgia, serif;
    font-size: 24px;
    color: #444444;
    margin-bottom: 35px;
}

/* Horizontal line */
hr {
    border: none;
    border-top: 1px solid #d4cec4;
    margin: 25px 0;
}

/* Text input label */
label {
    color: #111111 !important;
    font-weight: 700 !important;
    font-size: 17px !important;
}

/* Input box */
.stTextInput input {
    background-color: #f0efeb !important;
    color: #111111 !important;
    border: 1px solid #c8c2b8 !important;
    border-radius: 8px !important;
    padding: 15px !important;
    font-size: 18px !important;
}

/* Input focus */
.stTextInput input:focus {
    border: 1px solid #111111 !important;
    box-shadow: none !important;
}

/* Button */
.stButton > button {
    background-color: #111111 !important;
    color: #ffffff !important;
    border-radius: 8px !important;
    border: none !important;
    padding: 0.75rem 1.5rem !important;
    font-size: 18px !important;
    font-weight: 700 !important;
}

.stButton > button:hover {
    background-color: #333333 !important;
    color: #ffffff !important;
}

/* Result card */
.result-card {
    background-color: #f3f1ec;
    border: 1px solid #d2cbc0;
    border-radius: 10px;
    padding: 30px;
    margin-top: 30px;
    color: #111111;
}

.result-label {
    font-family: Georgia, serif;
    font-size: 22px;
    color: #222222;
    margin-bottom: 8px;
}

.result-word {
    font-family: Georgia, serif;
    font-size: 42px;
    font-weight: 800;
    color: #111111;
}

/* Footer */
.footer {
    text-align: center;
    color: #666666;
    font-family: Georgia, serif;
    margin-top: 80px;
    font-size: 16px;
}

</style>
""", unsafe_allow_html=True)


# ================= LOAD MODEL, TOKENIZER, MAX LEN =================
@st.cache_resource
def load_assets():
    model_path = "lstm_model.h5"
    tokenizer_path = "tokenizer.pkl"
    max_len_path = "max_len.pkl"

    if not os.path.exists(model_path):
        st.error("Model file not found: lstm_model.h5")
        st.stop()

    if not os.path.exists(tokenizer_path):
        st.error("Tokenizer file not found: tokenizer.pkl")
        st.stop()

    if not os.path.exists(max_len_path):
        st.error("Max length file not found: max_len.pkl")
        st.stop()

    model = tf.keras.models.load_model(model_path)

    with open(tokenizer_path, "rb") as file:
        tokenizer = pickle.load(file)

    with open(max_len_path, "rb") as file:
        max_len = pickle.load(file)

    return model, tokenizer, max_len


model, tokenizer, max_len = load_assets()


# ================= PREDICTION FUNCTION =================
def predict_next_word(seed_text):
    seed_text = seed_text.strip()

    if seed_text == "":
        return "Please enter a sentence."

    token_list = tokenizer.texts_to_sequences([seed_text])[0]

    if len(token_list) == 0:
        return "Word not found in vocabulary."

    # Convert max_len safely into integer
    input_length = int(max_len)

    # For next-word prediction models, input length is usually max_len - 1
    token_list = pad_sequences(
        [token_list],
        maxlen=input_length - 1,
        padding="pre"
    )

    prediction = model.predict(token_list, verbose=0)

    predicted_index = np.argmax(prediction, axis=1)[0]

    predicted_word = tokenizer.index_word.get(predicted_index, "Word not found")

    return predicted_word


# ================= SIDEBAR =================
with st.sidebar:
    st.markdown("## About")
    st.markdown("---")

    st.markdown("### Next Word Prediction App")
    st.write(
        "This application predicts the next possible word based on the input "
        "sentence using a trained deep learning model built with TensorFlow and Keras."
    )

    st.markdown("### ⚙️ Tech Stack")
    st.markdown("""
    - Python  
    - TensorFlow / Keras  
    - Natural Language Processing  
    - LSTM Neural Network  
    - Streamlit  
    """)

    st.markdown("### 🧠 Model Used")
    st.write("LSTM based neural network trained for sequence prediction.")

    st.markdown("---")

    st.markdown("### 🗂 Dataset")
    st.write("Trained on custom quote/text data.")


# ================= MAIN UI =================
# ================= MAIN UI =================
left_space, main_col, right_space = st.columns([1, 3, 1])

with main_col:
    st.markdown(
        '<div class="main-title">🪶 Next Word Prediction</div>',
        unsafe_allow_html=True
    )

    st.markdown("<hr>", unsafe_allow_html=True)

    st.markdown(
        '<div class="subtitle">Enter a sentence and predict the next possible word.</div>',
        unsafe_allow_html=True
    )

    user_input = st.text_input(
        "Type your sentence here",
        placeholder="Example: You are"
    )

    predict_button = st.button("Predict")

    if predict_button:
        next_word = predict_next_word(user_input)

        st.markdown(
            f"""
            <div class="result-card">
                <div class="result-label">✨ Predicted Next Word:</div>
                <div class="result-word">{next_word}</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    st.markdown("<hr>", unsafe_allow_html=True)

    st.markdown(
        '<div class="footer">Made with ❤️ using Streamlit & TensorFlow</div>',
        unsafe_allow_html=True
    )
