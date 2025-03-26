import pandas as pd
import joblib
import openmanus

def predict(test_df):
    model = joblib.load('model/sepsis_model.pkl')
    probabilities = model.predict_proba(test_df)[:, 1]  # Get probability scores
    predictions = (probabilities > 0.5).astype(int)

    results = pd.DataFrame({'Probability': probabilities, 'Prediction': predictions})
    results.to_csv('results/predictions.csv', index=False)

    # Log predictions to OpenManus
    experiment = openmanus.get_experiment()
    experiment.log_artifact("Predictions", "results/predictions.csv")

    print("Predictions saved to results/predictions.csv")
