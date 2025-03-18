from flask import Flask, request, jsonify
import pickle
import numpy as np
import pandas as pd

app = Flask(__name__)

# ✅ Load pre-trained models and encoders
with open('models/best_score_model.pkl', 'rb') as f:
    score_model = pickle.load(f)

with open('models/best_wickets_model.pkl', 'rb') as f:
    wickets_model = pickle.load(f)

with open('models/encoders.pkl', 'rb') as f:
    encoders = pickle.load(f)

# 🎯 Define features for prediction
score_features = ['batting_team', 'venue', 'toss_decision', 'powerplay_runs', 'recent_form']
wickets_features = ['batting_team', 'venue', 'toss_decision', 'powerplay_wickets', 'recent_form']


# ✅ Encode input features for models
def encode_input(data, features):
    encoded_data = []
    for col in features:
        if col in encoders and col in data:
            encoded_data.append(encoders[col].transform([data[col]])[0])
        else:
            encoded_data.append(data[col])
    return np.array(encoded_data).reshape(1, -1)


# 🎯 Predict Score Endpoint
@app.route('/predict_score', methods=['POST'])
def predict_score():
    try:
        data = request.get_json()

        # Validate input fields
        for field in score_features:
            if field not in data:
                return jsonify({'error': f'Missing field: {field}'}), 400

        input_data = encode_input(data, score_features)
        predicted_score = score_model.predict(input_data)[0]
        return jsonify({'predicted_score': round(predicted_score, 2)})

    except Exception as e:
        return jsonify({'error': str(e)}), 500


# 🎯 Predict Wickets Endpoint
@app.route('/predict_wickets', methods=['POST'])
def predict_wickets():
    try:
        data = request.get_json()

        for field in wickets_features:
            if field not in data:
                return jsonify({'error': f'Missing field: {field}'}), 400

        input_data = encode_input(data, wickets_features)
        predicted_wickets = wickets_model.predict(input_data)[0]
        return jsonify({'predicted_wickets': round(predicted_wickets, 2)})

    except Exception as e:
        return jsonify({'error': str(e)}), 500


# 🏏 Health Check Endpoint
@app.route('/', methods=['GET'])
def health_check():
    return jsonify({'status': 'API is running successfully!'})


# 🎯 Run Flask application
if __name__ == '__main__':
    app.run(debug=True)
