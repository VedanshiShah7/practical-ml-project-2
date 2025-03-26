import shap
import matplotlib.pyplot as plt
import joblib

def explain_predictions(model, data):
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(data)

    plt.figure()
    shap.summary_plot(shap_values, data, show=False)
    plt.savefig('results/shap_summary_plot.png')
