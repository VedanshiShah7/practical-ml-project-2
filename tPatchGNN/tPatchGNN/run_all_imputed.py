import os
import torch
import pandas as pd
import numpy as np
from model.tPatchGNN import tPatchGNN

# Load trained model checkpoint
CKPT_PATH = "experiments/experiment_48851.ckpt"
OUTPUT_FILE = "test_imputed.csv"

import torch
from argparse import Namespace

torch.serialization.add_safe_globals([Namespace])

def load_model(ckpt_path):
    checkpt = torch.load(ckpt_path, map_location=torch.device("cpu"), weights_only=False)
    ckpt_args = checkpt['args']
    state_dict = checkpt['state_dicts']

    model = tPatchGNN(ckpt_args)
    model.load_state_dict(state_dict)
    model.to("cpu")
    
    return model, ckpt_args

# Load the model
model, args = load_model(CKPT_PATH)

def impute_missing_values(model, row):
    observed_data = row.copy()
    mask = ~observed_data.isna()
    observed_data[~mask] = 0  # Replace NaNs with zeros for processing
    
    input_tensor = torch.tensor(observed_data.values, dtype=torch.float32).unsqueeze(0)
    mask_tensor = torch.tensor(mask.values, dtype=torch.float32).unsqueeze(0)
    
    # Debugging: print input tensor shape before reshaping
    print("input_tensor shape before reshape:", input_tensor.shape)
    
    # Handle different input tensor shapes
    if input_tensor.dim() == 1:
        input_tensor = input_tensor.view(1, -1, 1)  # Handle 1D case (flatten the tensor)
    elif input_tensor.dim() == 2:
        input_tensor = input_tensor.view(1, input_tensor.shape[0], 1, input_tensor.shape[1])  # Handle 2D case
    elif input_tensor.dim() == 3:
        # If the tensor has 3 dimensions, no reshaping needed
        pass
    else:
        raise ValueError(f"Unexpected tensor shape: {input_tensor.shape}")

    # Debugging: print reshaped input tensor shape
    print("input_tensor shape after reshape:", input_tensor.shape)

    with torch.no_grad():
        imputed_values = model.forecasting(None, input_tensor, None, mask_tensor)
    
    row[~mask] = imputed_values.squeeze().numpy()[~mask]
    return row

# Read train and test datasets
df_train = pd.read_csv("/Users/vedanshi/Documents/GitHub/practical-ml-project-2/main/data/train.csv")
df_test = pd.read_csv("/Users/vedanshi/Documents/GitHub/practical-ml-project-2/main/data/test.csv")

# Impute missing values in test dataset
imputed_test = df_test.apply(lambda row: impute_missing_values(model, row), axis=1)

# Save the imputed test dataset
imputed_test.to_csv(OUTPUT_FILE, index=False)
print(f"Imputed test data saved to {OUTPUT_FILE}")
