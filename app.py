from flask import Flask, render_template, request
import pickle
import numpy as np
import os
import traceback

app = Flask(__name__)

# Load model safely
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(BASE_DIR, "diabetes-prediction-rfc-model.pkl")

classifier = None

try:
    with open(model_path, "rb") as f:
        classifier = pickle.load(f)
    print("✅ Model loaded successfully")

except Exception as e:
    print("❌ Error loading model:")
    print(e)
    traceback.print_exc()


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    if classifier is None:
        return "Model failed to load. Check Render logs."

    try:
        preg = int(request.form['pregnancies'])
        glucose = int(request.form['glucose'])
        bp = int(request.form['bloodpressure'])
        st = int(request.form['skinthickness'])
        insulin = int(request.form['insulin'])
        bmi = float(request.form['bmi'])
        dpf = float(request.form['dpf'])
        age = int(request.form['age'])

        data = np.array([[preg, glucose, bp, st, insulin, bmi, dpf, age]])
        prediction = classifier.predict(data)

        return render_template('result.html', prediction=prediction[0])

    except Exception as e:
        return f"Prediction Error: {str(e)}"


if __name__ == '__main__':
    app.run(debug=True)
