import pandas as pd
import joblib

def predict(df, model):
    probabilities = model.predict_proba(df)[:, 1]
    predictions = (probabilities > 0.5).astype(int)
    results = pd.DataFrame({'Probability': probabilities, 'Prediction': predictions})
    results.to_csv('results/predictions.csv', index=False)
    return results
