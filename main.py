import streamlit as st
import requests

# import os
# os.environ['PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION'] = 'python'

# flask --app api.py run --port=5000
prediction_endpoint = "http://127.0.0.1:5000/predict"

st.title("Text Sentiment Predictor")

user_input = st.text_input("Enter text and click on Predict", "")

if st.button("Predict"):
        response = requests.post(prediction_endpoint, data={"text": user_input})
        response = response.json()
        st.write(f"Predicted sentiment: {response['prediction']}")
