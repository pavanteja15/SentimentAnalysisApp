
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 24 09:21:12 2025

@author: Lenovo
"""
from flask import Flask, render_template, request, redirect, url_for
import joblib
import smtplib
from email.message import EmailMessage

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

@app.route('/contact')
def contact():
    return render_template('contact.html')

@app.route('/send_message', methods=['POST'])
def send_message():
    name = request.form['name']
    email = request.form['email']
    message = request.form['message']
    text = request.form['text']  # Input text from the user
    sentiment = request.form['sentiment']  # Predicted sentiment

    # Include user input and sentiment in the email message
    try:
        msg = EmailMessage()
        msg.set_content(
            f"Name: {name}\n"
            f"Email: {email}\n"
            f"Message: {message}\n\n"
            f"User Input: {text}\n"
            f"Predicted Sentiment: {sentiment}"
        )
        msg['Subject'] = 'Feedback Submission for Sentiment Analysis App'
        msg['From'] = email
        msg['To'] = 'pavanteja1515@gmail.com'

        with smtplib.SMTP('smtp.gmail.com', 587) as server:
            server.starttls()
            server.login('pavanteja1515@gmail.com', 'pcnf vbui cemr jkpa')  # Replace with your credentials
            server.send_message(msg)
        
        # Render the result page with a success message
        return render_template('result.html', text=text, sentiment=sentiment, bg_color="white", feedback_submitted=True)
    except Exception as e:
        return f"An error occurred: {e}"

if __name__ == '__main__':
    app.run(debug=True)
