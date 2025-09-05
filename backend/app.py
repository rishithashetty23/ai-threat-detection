from flask import Flask, request, jsonify
import joblib
import pandas as pd
import numpy as np

app = Flask(__name__)

# Load the trained model
try:
    model_data = joblib.load('cicids_model.pkl')
    model = model_data['model']
    feature_names = model_data['feature_names']
    print("Model loaded successfully!")
    print(f"Expected features: {feature_names}")
except FileNotFoundError:
    print("Model file not found. Please train the model first.")
    model = None
    feature_names = []

@app.route('/detect', methods=['POST'])
def detect_threat():
    """
    API endpoint for threat detection
    """
    if model is None:
        return jsonify({'error': 'Model not loaded'}), 500
    
    try:
        data = request.json
        features = data.get('features')
        
        if not features:
            return jsonify({'error': 'No features provided'}), 400
        
        if len(features) != len(feature_names):
            return jsonify({
                'error': f'Expected {len(feature_names)} features, got {len(features)}'
            }), 400
        
        # Create DataFrame with proper feature names
        input_df = pd.DataFrame([features], columns=feature_names)
        
        # Make prediction
        prediction = model.predict(input_df)[0]
        probability = model.predict_proba(input_df)[0]
        
        result = {
            'prediction': int(prediction),
            'result': 'Threat detected' if prediction == 1 else 'No threat detected',
            'confidence': {
                'benign': float(probability[0]),
                'threat': float(probability[1])
            }
        }
        
        return jsonify(result)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({'status': 'healthy', 'model_loaded': model is not None})

if __name__ == '__main__':
    print("Starting AI Threat Detection API...")
    print("API will be available at: http://localhost:5000")
    app.run(debug=True, port=5000)
