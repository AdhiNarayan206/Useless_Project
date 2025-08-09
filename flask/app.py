
import os
from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import numpy as np
from sklearn.preprocessing import LabelEncoder



app = Flask(__name__)
CORS(app)

# Always load model/encoder relative to this file's directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model = joblib.load(os.path.join(BASE_DIR, "chip_predictor_model.pkl"))
le = joblib.load(os.path.join(BASE_DIR, "flavor_encoder.pkl"))


# Accept both GET and POST for easier testing
@app.route('/predict', methods=['GET', 'POST'])
def predict():
    """
    Accepts input as JSON (POST) or query params (GET):
      - flavor: string
      - weight_g: float or int
    Returns: JSON { predicted: int }
    """
    if request.method == 'POST':
        data = request.get_json()
        if not data or 'flavor' not in data or 'weight_g' not in data:
            return jsonify({"error": "Missing required fields (flavor, weight_g)"}), 400
        flavor = data['flavor']
        weight = float(data['weight_g'])
    else:
        # GET method: use query params
        flavor = request.args.get('flavor')
        weight = request.args.get('weight_g')
        if not flavor or not weight:
            return jsonify({"error": "Missing required fields (flavor, weight_g)"}), 400
        try:
            weight = float(weight)
        except Exception:
            return jsonify({"error": "weight_g must be a number"}), 400

    # Encode flavor
    try:
        flavor_encoded = le.transform([flavor])[0]
    except Exception:
        return jsonify({"error": f"Unknown flavor: {flavor}"}), 400
    input_data = [[weight, flavor_encoded]]

    # Predict
    prediction = model.predict(input_data)[0]
    prediction = int(prediction)

    return jsonify({"predicted": prediction})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)