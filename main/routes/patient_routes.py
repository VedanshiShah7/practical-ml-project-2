from flask import Blueprint, request, jsonify
from models.tpatch_gnn_model import predict_values
import csv
import numpy as np

patient_bp = Blueprint('patient_bp', __name__)

# Load patient data from CSV
def load_patient_data(csv_file):
    patient_data = []
    
    with open(csv_file, mode='r', encoding='utf-8') as f:
        reader = csv.DictReader(f)  # Reads CSV into a list of dictionaries
        
        for row in reader:
            if "icustayid" in row:
                row["id"] = int(float(row["icustayid"]))  # Convert to int
            else:
                raise KeyError("CSV file must have 'icustayid' as the patient ID.")
            
            # Convert numerical values where applicable
            row = convert_patient_data(row)
            patient_data.append(row)

    # Impute missing values
    patient_data = impute_missing_values(patient_data)

    return patient_data


def convert_patient_data(patient):
    """Convert numerical values and handle missing data."""
    for key, value in patient.items():
        if value is None or value == "" or (isinstance(value, str) and value.lower() == "null"):
            patient[key] = None  # Treat empty or 'null' as missing
        elif isinstance(value, str) and value.replace('.', '', 1).isdigit():  # Convert floats
            patient[key] = float(value)
        elif isinstance(value, str) and value.isdigit():  # Convert integers
            patient[key] = int(value)
    
    return patient


def impute_missing_values(patient_data):
    """Fill missing numerical values with column mean and categorical with 'Unknown'."""
    if not patient_data:
        return patient_data  # Return empty list if no data

    # Convert to list of dictionaries for processing
    numeric_columns = {}  # Store numeric values for mean calculation

    # Step 1: Collect numeric column values
    for patient in patient_data:
        for key, value in patient.items():
            if isinstance(value, (int, float)):
                if key not in numeric_columns:
                    numeric_columns[key] = []
                numeric_columns[key].append(value)

    # Step 2: Calculate mean for numeric columns
    column_means = {key: np.mean(values) for key, values in numeric_columns.items()}

    # Step 3: Fill missing values
    for patient in patient_data:
        for key, value in patient.items():
            if value is None:
                if key in column_means:  # Numeric column
                    patient[key] = column_means[key]
                else:  # Categorical column
                    patient[key] = "Unknown"

    return patient_data


# Load patient data from test.csv
csv_file_path = "/Users/vedanshi/Documents/GitHub/practical-ml-project-2/main/data/test.csv"
patient_data = load_patient_data(csv_file_path)


# 🔍 Search for patients by `icustayid`
@patient_bp.route("/search", methods=["GET"])
def search_patient():
    query = request.args.get("query", "").lower()
    results = [p for p in patient_data if str(p["id"]) == query]
    return jsonify(results)


# 🏥 Get patient profile and impute missing values
@patient_bp.route("/patient/<int:patient_id>", methods=["GET"])
def get_patient(patient_id):
    patient = next((p for p in patient_data if p["id"] == patient_id), None)

    if not patient:
        return jsonify({"error": "Patient not found"}), 404

    # Predict and fill missing values using tPatchGNN
    imputed_data = predict_values(patient)
    return jsonify(imputed_data)
