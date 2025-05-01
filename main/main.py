import os
import openmanus
from preprocessing import load_data, preprocess_data
from model import train_model
from explain import explain_predictions
from inference import predict

if __name__ == "__main__":
    os.makedirs('model', exist_ok=True)
    os.makedirs('results', exist_ok=True)

    # Initialize OpenManus for experiment tracking
    experiment = openmanus.Experiment(name="mortality-prediction", project="sepsis-ehr-analysis")

    # Load and preprocess data
    train_df, test_df = load_data("data/train.csv", "data/test.csv")
    train_df, test_df = preprocess_data(train_df, test_df)

    # Train model and log results
    model = train_model(train_df)

    # Log metrics and visualizations with OpenManus
    experiment.log_metrics({"train_size": len(train_df), "test_size": len(test_df)})

    # Explain model predictions and log SHAP plot
    explain_predictions(model, train_df.sample(100))

    # Log SHAP summary plot to OpenManus dashboard
    experiment.log_artifact("SHAP Summary Plot", "results/shap_summary_plot.png")

    # Make predictions on test set and log the results
    predict(test_df)
    experiment.log_artifact("Predictions", "results/predictions.csv")

    # Finish the OpenManus experiment
    experiment.finish()
