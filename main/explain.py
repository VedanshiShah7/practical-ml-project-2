import shap
import matplotlib.pyplot as plt
import sys
import os

# Add OpenManus path to Python's module search path
openmanus_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../openmanus"))
sys.path.append(openmanus_path)

import openmanus

def explain_predictions(model, data):
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(data)

    plt.figure()
    shap.summary_plot(shap_values, data, show=False)
    plt.savefig('results/shap_summary_plot.png')

    # Log the SHAP plot to OpenManus
    experiment = openmanus.get_experiment()
    experiment.log_artifact("SHAP Summary Plot", "results/shap_summary_plot.png")
