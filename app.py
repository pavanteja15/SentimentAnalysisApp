# -*- coding: utf-8 -*-
"""
Created on Fri Jan 24 09:21:12 2025

@author: Lenovo
"""
from flask import Flask, render_template, request
import joblib

app = Flask(__name__)

# Load the trained components
vectorizer = joblib.load('vectorizer.pkl')
selector = joblib.load('selector.pkl')
model = joblib.load('svm_sentiment_model.pkl')

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
    sentiment_code = model.predict(selected_text)[0]
    sentiment = "Positive" if sentiment_code == 1 else "Neutral" if sentiment_code == 0 else "Negative"
    
    # Dynamic background color for sentiments
    bg_color = "green" if sentiment == "Positive" else "yellow" if sentiment == "Neutral" else "red"
    
    # Render the result page with variables
    return render_template('result.html', text=text, sentiment=sentiment, bg_color=bg_color)

if __name__ == '__main__':
    app.run(debug=True)
