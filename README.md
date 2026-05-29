# 🪶 Next Word Prediction using LSTM

A deep learning-powered Natural Language Processing (NLP) application that predicts the most probable next word for a given text sequence using an LSTM (Long Short-Term Memory) neural network.

Built with TensorFlow/Keras and deployed through a Streamlit web interface, this project demonstrates sequence modeling, text preprocessing, and real-time inference for language prediction tasks.

---

## 🚀 Features

* Predicts the next likely word from an input text sequence
* Interactive Streamlit web interface
* LSTM-based deep learning architecture
* Real-time inference and prediction
* Clean and intuitive user experience
* Serialized model and tokenizer for efficient deployment

---

## 🧠 Model Architecture

The model is built using:

* Embedding Layer
* LSTM Layer
* Dense Output Layer with Softmax Activation

The network learns contextual relationships between words and predicts the most probable next token based on previously observed sequences.

---

## ⚙️ Tech Stack

**Programming Language**

* Python

**Machine Learning / Deep Learning**

* TensorFlow
* Keras
* NumPy

**Natural Language Processing**

* Tokenization
* Sequence Generation
* Text Vectorization

**Web Application**

* Streamlit

**Version Control**

* Git
* GitHub

---

## 📂 Project Structure

```text
next_word_pred/
│
├── app.py
├── lstm_model.h5
├── tokenizer.pkl
├── max_len.pkl
├── requirements.txt
├── README.md
└── .gitignore
```

---

## 🔍 How It Works

1. User enters a text sequence.
2. Text is tokenized using the trained tokenizer.
3. Sequence is padded to the required input length.
4. The LSTM model processes the sequence.
5. The most probable next word is predicted and displayed.

---

## 📸 Application Preview


![![App Screenshot](screenshot1.png<img width="1890" height="848" alt="screenshot1 png" src="https://github.com/user-attachments/assets/a068e5ef-8ecd-4522-88da-9af6218a051b" />
)]

![![App Screenshot](screenshot2.png)]<img width="1894" height="818" alt="screenshot2 png" src="https://github.com/user-attachments/assets/f07098bc-4743-4916-b01e-657f5f5ab80e" />

---

## 🛠️ Installation

Clone the repository:

```bash
git clone https://github.com/AyushChaki/next_word_pred.git
cd next_word_pred
```

Create a virtual environment:

```bash
python -m venv venv
```

Activate the environment:

```bash
venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the application:

```bash
streamlit run app.py
```

---

## 🎯 Future Improvements

* Top-N word prediction
* Transformer-based language models
* Improved text preprocessing pipeline
* Larger training corpus
* Enhanced UI and model explainability

---

## 👨‍💻 Author

**Ayush Chaki**

Undergraduate Student, NIT Rourkela

Interested in:

* Machine Learning
* Deep Learning
* Natural Language Processing
* Data Science
* AI Applications
