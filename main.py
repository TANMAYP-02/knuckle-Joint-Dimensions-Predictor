from flask import Flask, request, jsonify
import numpy as np
from flask_cors import CORS
import joblib

app = Flask(__name__)
CORS(app)  # allow all origins

# Load trained model
model = joblib.load("./rf_knuckle_joint_model.pkl")
cols_y = ['d1', 'd2', 'd3', 't', 't1', 't2']

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json() 
        load = float(data['load'])
        fos_pin = float(data['fos_pin']) 
        fos_eye = float(data['fos_eye'])
        print("Inputs:", load, fos_pin, fos_eye)
    except (KeyError, ValueError, TypeError):
        return jsonify({'error': 'Invalid input'}), 400

    arr = np.array([[load, fos_pin, fos_eye]])
    pred = model.predict(arr)[0]
    result = {col: round(val, 2) for col, val in zip(cols_y, pred)}
    return jsonify(result)

if __name__ == "__main__":
    app.run(debug=True)
