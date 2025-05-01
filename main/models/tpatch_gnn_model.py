import sys
import os
import subprocess
# Add the correct path to tPatchGNN
tpatch_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../t-PatchGNN/tPatchGNN"))
sys.path.insert(0, tpatch_path)

print(f"✅ Added tPatchGNN to PYTHONPATH: {tpatch_path}")

# Now import tPatchGNN
from tPatchGNN import tPatchGNN

# Load the tPatchGNN model
def load_model(model_id=48851):
    model_path = os.path.join(os.path.dirname(__file__), "../t-PatchGNN/tPatchGNN/run_samples.py")
    command = ["python", model_path, "--load", str(model_id)]
    result = subprocess.run(command, capture_output=True, text=True)
    
    if result.returncode != 0:
        raise RuntimeError(f"Error loading t-PatchGNN model: {result.stderr}")
    
    print(f"Model {model_id} loaded successfully!")
    return result.stdout


# Function to predict missing values for a patient
def predict_values(patient_data, model_id=48851):
    # Run t-PatchGNN to get imputed results
    model_output = load_model(model_id)
    
    # Extract and process the prediction results
    predictions = parse_predictions(model_output, patient_data)
    return predictions


def parse_predictions(model_output, patient_data):
    # Parse predicted results from run_samples.py output
    parsed_results = {}
    for line in model_output.splitlines():
        if "Predicted" in line:
            key, value = line.split(": ")
            parsed_results[key.strip()] = float(value.strip())
    
    # Merge with original data and replace missing values
    for key, value in parsed_results.items():
        if patient_data[key] is None or patient_data[key] == 0:
            patient_data[key] = value
    
    return patient_data
