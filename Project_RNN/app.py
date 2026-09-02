import tensorflow as tf
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import pad_sequences

from tensorflow.keras.datasets import imdb

word_index=imdb.get_word_index()
reversed_word_index=[]
for i,j in enumerate(word_index):
    reversed_word_index.append((j,i))

model=load_model("simple_rnn_imdb.h5")

def preprocess_text(text):
    words = text.lower().split()
    encoded_review = [word_index.get(word, 2) + 3 for word in words]
    padded_review = pad_sequences([encoded_review], maxlen=500)
    return padded_review

def predict_review(text):
    proproceesed_input=preprocess_text(text)
    prediction=model.predict(proproceesed_input)
    sentiment="Positive" if prediction[0][0] > 0.5 else "Negative"
    return sentiment,prediction[0][0]

import streamlit as st


st.title('IMDB Movie Review Sentiment Analysis')
st.write('Enter a movie review to classify it as positive or negative.')

# User input
user_input = st.text_area('Movie Review')

if st.button('Classify'):

    preprocessed_input = preprocess_text(user_input)

    ## Make prediction
    prediction = model.predict(preprocessed_input)

    sentiment = 'Positive' if prediction[0][0] > 0.5 else 'Negative'
    
    # Display the result
    st.write(f'Sentiment: {sentiment}')
    st.write(f'Prediction Score: {prediction[0][0]}')

else:
    st.write('Please enter a movie review.')