import streamlit as st
import numpy as np
import pickle
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

model=load_model("hamilton.h5")

with open("tokenizer.pickle","rb") as handle:
    tokenizer=pickle.load(handle)

def prediction_next_word(model, tokenizer, text, max_sequence_len):
    token_list = tokenizer.texts_to_sequences([text])[0]

    if len(token_list) >= max_sequence_len:
        token_list = token_list[-(max_sequence_len - 1):]

    token_list = pad_sequences(
        [token_list],
        maxlen=max_sequence_len - 1,
        padding="pre")
    
    predicted = model.predict(token_list, verbose=0)
    predicted_word_index = np.argmax(predicted, axis=1)[0]
    for word, index in tokenizer.word_index.items():
        if index == predicted_word_index:
            return word
    return None

def complete_sentence(model, tokenizer, text, max_sequence_len, num_words):
    result = text
    for _ in range(num_words):
        next_word = prediction_next_word(model, tokenizer, result, max_sequence_len)
        if next_word is None:
            break
        result += " " + next_word
    return result

st.title("Next Word and Sentences Prediction With LSTM and Early Stopping")
input_text=st.text_input("Enter the sequence of words")
if st.button("Predict"):
    max_sequence_len=model.input_shape[1]+1
    next_word=prediction_next_word(model,tokenizer,input_text,max_sequence_len)
    complete_sentences=complete_sentence(model, tokenizer, input_text, max_sequence_len, num_words=5)
    st.write(f"Next word:{next_word}")
    st.write(f"Next sentence:{complete_sentences}")