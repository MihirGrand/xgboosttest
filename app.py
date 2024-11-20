from flask import Flask, request, jsonify, render_template
import pickle
import numpy as np

app = Flask(__name__)

# Load the model
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

# Start the server on a specific host and port
if __name__ == '__main__':
    # Use 0.0.0.0 to allow the server to be accessed externally
    port = 5000  # Default port
    app.run(host='0.0.0.0', port=port, debug=True)

