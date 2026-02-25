from flask import Flask, render_template, request
import pickle
import numpy as np
import os

app = Flask(__name__)

# Load model safely
try:
    model_path = os.path.join(os.getcwd(), "diabetes-prediction-rfc-model.pkl")
    with open(model_path, "rb") as f:
        classifier = pickle.load(f)
    print("Model loaded successfully")
except Exception as e:
    print("Error loading model:", e)
    classifier = None


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST' and classifier is not None:
        preg = int(request.form['pregnancies'])
        glucose = int(request.form['glucose'])
        bp = int(request.form['bloodpressure'])
        st = int(request.form['skinthickness'])
        insulin = int(request.form['insulin'])
        bmi = float(request.form['bmi'])
        dpf = float(request.form['dpf'])
        age = int(request.form['age'])

        data = np.array([[preg, glucose, bp, st, insulin, bmi, dpf, age]])
        my_prediction = classifier.predict(data)

        return render_template('result.html', prediction=my_prediction)

    return "Model not loaded properly"


if __name__ == '__main__':
    app.run()
