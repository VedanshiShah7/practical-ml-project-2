import pandas as pd
import numpy as np
from sklearn.experimental import enable_iterative_imputer  # Enable before importing
from sklearn.impute import IterativeImputer
from sklearn.preprocessing import StandardScaler

def load_data(train_path, test_path):
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    return train_df, test_df

def handle_missing_data(df):
    cols_with_zero_missing = ['Weight_kg', 'GCS', 'HR', 'SysBP', 'MeanBP', 'DiaBP']
    df[cols_with_zero_missing] = df[cols_with_zero_missing].replace(0, np.nan)

    # Use IterativeImputer for better imputation
    imputer = IterativeImputer(random_state=42)
    df[df.columns] = imputer.fit_transform(df)

    return df


def aggregate_time_series(df, is_train=True):
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
        agg_funcs['mortality_90d'] = 'first'  # ✅ Ensure we keep mortality_90d

    df = df.groupby('icustayid').agg(agg_funcs)
    df.columns = ['_'.join(col).strip() for col in df.columns.values]  # Flatten multi-index
    df.reset_index(inplace=True)

    return df


def preprocess_data(train_df, test_df):
    train_df = handle_missing_data(train_df)
    test_df = handle_missing_data(test_df)

    train_df = aggregate_time_series(train_df, is_train=True)
    test_df = aggregate_time_series(test_df, is_train=False)

    # Normalize numerical columns
    scaler = StandardScaler()
    num_cols = train_df.select_dtypes(include=['float64', 'int64']).columns

    # Find the correct column name for mortality
    mortality_col = 'mortality_90d_first' if 'mortality_90d_first' in num_cols else 'mortality_90d'
    
    # Only drop the mortality column if it exists
    if mortality_col in num_cols:
        num_cols = num_cols.drop(mortality_col)

    train_df[num_cols] = scaler.fit_transform(train_df[num_cols])
    test_df[num_cols] = scaler.transform(test_df[num_cols])

    return train_df, test_df
