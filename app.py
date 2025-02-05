from flask import Flask, request, jsonify, render_template
import pickle
import numpy as np
import string
from nltk.corpus import stopwords
import nltk
nltk.download('punkt')
from nltk.stem.porter import PorterStemmer
ps = PorterStemmer()

# Load the trained model
model_path = 'model.pkl'
with open(model_path, 'rb') as file:
    model = pickle.load(file)
model_path1 = 'vectorizer.pkl'
with open(model_path1,'rb') as file:
    tfidf = pickle.load(file)

app = Flask(__name__)
def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)
    # Text is converted in to list
    len(text)
    print(text)
    y = []
    for i in text:
        if i.isalnum():
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)

    text = y[:]
    y.clear()
    for i in text:
        y.append(ps.stem(i))

    return " ".join(y)

@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    # Extract data from form
    input_sms = request.form.get("Input_SMS")
    transformed_sms = transform_text(input_sms)
    vector_input = tfidf.transform([transformed_sms])

    # Make prediction
    prediction = model.predict(vector_input)
    output = 'Spam' if prediction[0] == 1 else 'Not Spam'
    #output = input_sms

    return render_template('index.html', prediction_text='Prediction: {}'.format(output))


if __name__ == "__main__":
    app.run(debug=True)