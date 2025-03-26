from flask import Flask, request, jsonify, send_file
import pandas as pd
import joblib
import shap
import matplotlib.pyplot as plt
import os
from inference import predict
from explain import explain_predictions

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = "uploads"
os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)
os.makedirs("results", exist_ok=True)

# Load model
model = joblib.load("model/sepsis_model.pkl")

@app.route("/upload", methods=["POST"])
def upload_file():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400
    
    file = request.files["file"]
    filepath = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
    file.save(filepath)
    
    df = pd.read_csv(filepath)
    results = predict(df, model)
    explain_predictions(model, df.sample(100))  # Generate SHAP plot
    
    return jsonify({"predictions": results.to_dict(), "shap_plot": "results/shap_summary_plot.png"})

@app.route("/download", methods=["GET"])
def download_results():
    return send_file("results/predictions.csv", as_attachment=True)

if __name__ == "__main__":
    app.run(debug=True)
