from flask import Flask, request, jsonify
import pickle
import numpy as np

app = Flask(__name__)

with open('xgb_model.pkl', 'rb') as file:
    model = pickle.load(file)

@app.route('/')
def home():
    return "Hello, welcome to the Priority Prediction API!"

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get input data from the request
        data = request.get_json()
        features = np.array([data['illness_severity'], data['age'], data['transmittable'], data['disabled']]).reshape(1, -1)
        prediction = model.predict(features)
        # Round off the prediction to the nearest integer
        rounded_priority = round(prediction[0])
        return jsonify({'priority': int(rounded_priority)})
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)
