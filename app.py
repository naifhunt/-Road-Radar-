import pandas as pd
from flask import Flask, request, jsonify, render_template
import pickle
import numpy as np

app = Flask(__name__)

# Load the model and scaler
model = pickle.load(open("model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get the input data from the request
        data = request.get_json(force=True)



        # Extract the features
        features = [data['Date'], data['Day of the week'], data['CarCount'], data['BikeCount'],data['BusCount'], data['TruckCount'], data['Total'], data['hour'],data['minute'], data['AM/PM']]
        features = np.array(features).reshape(1, -1)  # Convert to 2D array

        # Scale the features
        scaled_features = scaler.transform(features)

        # Predict the class
        prediction = model.predict(scaled_features)
        output = prediction[0]

        return jsonify({'prediction': output})
    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True)