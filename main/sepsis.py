from model import train_model
import joblib
import pandas as pd

# Load and preprocess data
train_df = pd.read_csv("data/train.csv")
model = train_model(train_df)

# Save trained model
joblib.dump(model, "model/sepsis_model.pkl")
print("Model saved as model/sepsis_model.pkl")
