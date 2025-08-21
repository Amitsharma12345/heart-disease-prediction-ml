from flask import Flask, request, render_template
import pickle
import numpy as np

app = Flask(__name__)

# Model & scaler load
model = pickle.load(open('Amit.pkl', 'rb'))
scaler = pickle.load(open('sharma.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    inputs = [float(x) for x in request.form.values()]
    final_input = scaler.transform([inputs])
    prediction = model.predict(final_input)

    result = 'Heart Disease Detected' if prediction[0] == 1 else 'No Heart Disease'

    return render_template('index.html', prediction_text=f'Prediction: {result}')

if __name__ == "__main__":
    app.run(debug=True)
