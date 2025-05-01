import os
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
from sklearn.experimental import enable_iterative_imputer  # noqa
from sklearn.impute import IterativeImputer
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE

from app.tool.base import BaseTool, ToolResult


class MortalityPredictionTool(BaseTool):
    name: str = "mortality_prediction"
    description: str = (
        "Run a complete sepsis mortality prediction pipeline. "
        "Loads and preprocesses EHR data, trains a model using SMOTE-balanced data, "
        "and generates predictions saved to a CSV file."
    )
    parameters: dict = {
        "type": "object",
        "properties": {
            "train_path": {
                "type": "string",
                "description": "Path to the training CSV file.",
            },
            "test_path": {
                "type": "string",
                "description": "Path to the testing CSV file.",
            },
        },
        "required": ["train_path", "test_path"],
    }

    async def execute(self, **kwargs) -> ToolResult:
        try:
            train_path = kwargs.get("train_path")
            test_path = kwargs.get("test_path")

            if not train_path or not test_path:
                return ToolResult(error="Both 'train_path' and 'test_path' are required.")

            # Create necessary folders
            os.makedirs("model", exist_ok=True)
            os.makedirs("results", exist_ok=True)

            # Load and preprocess data
            train_df, test_df = self.load_data(train_path, test_path)
            train_df, test_df = self.preprocess_data(train_df, test_df)

            # Train and save model
            model = self.train_model(train_df)

            # Run inference and save predictions
            self.predict(test_df)

            return ToolResult(output="Sepsis prediction pipeline executed successfully.")

        except Exception as e:
            return ToolResult(error=f"An error occurred: {str(e)}")

    def load_data(self, train_path: str, test_path: str):
        train_df = pd.read_csv(train_path)
        test_df = pd.read_csv(test_path)
        return train_df, test_df

    def handle_missing_data(self, df: pd.DataFrame):
        cols_with_zero_missing = ['Weight_kg', 'GCS', 'HR', 'SysBP', 'MeanBP', 'DiaBP']
        df[cols_with_zero_missing] = df[cols_with_zero_missing].replace(0, np.nan)
        imputer = IterativeImputer(random_state=42)
        df[df.columns] = imputer.fit_transform(df)
        return df

    def aggregate_time_series(self, df: pd.DataFrame, is_train: bool = True):
        agg_funcs = {
            'age': 'first',
            'gender': 'first',
            'elixhauser': 'first',
            'HR': ['mean', 'max', 'min'],
            'SysBP': ['mean', 'max', 'min'],
            'DiaBP': ['mean', 'max', 'min'],
            'MeanBP': ['mean', 'max', 'min'],
            'RR': ['mean', 'max', 'min'],
            'SpO2': ['mean', 'min'],
            'Temp_C': ['mean', 'max', 'min'],
            'SOFA': 'max',
            'SIRS': 'max',
        }

        if is_train and 'mortality_90d' in df.columns:
            agg_funcs['mortality_90d'] = 'first'

        df = df.groupby('icustayid').agg(agg_funcs)
        df.columns = ['_'.join(col).strip() for col in df.columns.values]
        df.reset_index(inplace=True)
        return df

    def preprocess_data(self, train_df: pd.DataFrame, test_df: pd.DataFrame):
        train_df = self.handle_missing_data(train_df)
        test_df = self.handle_missing_data(test_df)

        train_df = self.aggregate_time_series(train_df, is_train=True)
        test_df = self.aggregate_time_series(test_df, is_train=False)

        scaler = StandardScaler()
        num_cols = train_df.select_dtypes(include=['float64', 'int64']).columns
        mortality_col = 'mortality_90d_first' if 'mortality_90d_first' in num_cols else 'mortality_90d'

        if mortality_col in num_cols:
            num_cols = num_cols.drop(mortality_col)

        train_df[num_cols] = scaler.fit_transform(train_df[num_cols])
        test_df[num_cols] = scaler.transform(test_df[num_cols])

        return train_df, test_df

    def train_model(self, train_df: pd.DataFrame):
        mortality_col = 'mortality_90d_first' if 'mortality_90d_first' in train_df.columns else 'mortality_90d'

        X = train_df.drop(columns=[mortality_col])
        y = train_df[mortality_col]

        smote = SMOTE(random_state=42)
        X, y = smote.fit_resample(X, y)

        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

        model = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42)
        model.fit(X_train, y_train)

        y_pred = model.predict(X_val)
        f1 = f1_score(y_val, y_pred)
        print(f"Validation F1 Score: {f1:.4f}")

        joblib.dump(model, "model/sepsis_model.pkl")
        return model

    def predict(self, test_df: pd.DataFrame):
        model = joblib.load("model/sepsis_model.pkl")
        probabilities = model.predict_proba(test_df)[:, 1]
        predictions = (probabilities > 0.5).astype(int)

        results = pd.DataFrame({
            "Probability": probabilities,
            "Prediction": predictions
        })

        results.to_csv("results/predictions.csv", index=False)
        print("Predictions saved to results/predictions.csv")
