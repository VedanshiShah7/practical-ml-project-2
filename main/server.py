from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import joblib
from explain import explain_predictions
from inference import predict
import os

app = Flask(__name__)
CORS(app)  # Enable CORS to allow requests from your React app

# Load the trained model
model = joblib.load('model/sepsis_model.pkl')

@app.route('/predict', methods=['POST'])
def handle_predict():
    # Get file from frontend
    file = request.files['file']
    df = pd.read_csv(file)

    # Get predictions and SHAP explanation
    predictions = predict(df)
    explain_predictions(model, df.sample(100))

    # Return results (predictions and SHAP image path)
    return jsonify({
        'predictions': predictions.to_dict(orient='records'),
        'shap_plot': 'results/shap_summary_plot.png'
    })

if __name__ == "__main__":
    os.makedirs('results', exist_ok=True)
    app.run(debug=True, host='0.0.0.0', port=5000)
