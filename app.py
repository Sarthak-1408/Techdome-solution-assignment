import streamlit as st
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
import pickle
import re
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS


# Load the saved model and tokenizer
model = load_model('sentiment_analysis_model.h5')
with open('tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)

def preprocess_text(text):
    # Remove symbols including '<'
    text = re.sub(r'[^A-Za-z\s]', '', text)
    # Remove occurrences of '<<'
    text = text.replace('<<', '')
    # Remove integers
    text = re.sub(r'\b\d+\b', '', text)
    # Tokenize the text using split()
    tokens = text.split()
    # Remove stopwords and convert to lowercase
    tokens = [token.lower() for token in tokens if token.isalpha() and token.lower() not in ENGLISH_STOP_WORDS]
    return ' '.join(tokens)

# Streamlit UI
st.title("Techdome Solutions Private Limited")
st.header("AI Engineer Assignment")
st.subheader('Sentiment Analysis Web App')

# Text input for user to enter a review
review = st.text_area('Enter your review here:', '')

# Button to trigger sentiment analysis
if st.button('Predict Sentiment'):
    if review:
        # Preprocess the entered text
        review = preprocess_text(review)

        # Tokenize and pad the text
        review_seq = tokenizer.texts_to_sequences([review])
        review_pad = pad_sequences(review_seq, maxlen=100)

        # Make prediction
        prediction_prob = model.predict(review_pad)
        prediction = "Positive" if prediction_prob > 0.5 else "Negative"

        # Display the result
        st.write(f"{prediction} Sentiment, Confidence: {prediction_prob[0,0]:.4f}")
    else:
        st.warning('Please enter a review before predicting.')


