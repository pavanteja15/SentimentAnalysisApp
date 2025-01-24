# -*- coding: utf-8 -*-
"""
Created on Fri Jan 24 09:21:12 2025

@author: Lenovo
"""
from flask import Flask, render_template, request
import joblib

app = Flask(__name__)
app.secret_key = 'your_secret_key' 

# Load the trained components
vectorizer = joblib.load('C:/Users/Lenovo/Downloads/Sentiment Analysis Pro/vectorizer.pkl')
selector = joblib.load('C:/Users/Lenovo/Downloads/Sentiment Analysis Pro/selector.pkl')
model = joblib.load('C:/Users/Lenovo/Downloads/Sentiment Analysis Pro/svm_sentiment_model.pkl')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get the text from the form
    text = request.form['text']
    
    # Preprocess the input text
    vectorized_text = vectorizer.transform([text])
    selected_text = selector.transform(vectorized_text)
    
    # Predict sentiment
    sentiment = model.predict(selected_text)[0]
    
    # Render the result page with sentiment
    return render_template('result.html', text=text, sentiment=sentiment)

if __name__ == '__main__':
    app.run(debug=True)

