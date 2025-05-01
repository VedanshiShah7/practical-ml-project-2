import pandas as pd
import torch
from tPatchGNN import tPatchGNN  # Ensure you have T-Patch-GNN installed

def impute_missing_values(input_csv, output_csv):
    # Load CSV
    df = pd.read_csv(input_csv)

    # Identify missing values
    missing_mask = df.isnull()

    # Convert to tensor
    data_tensor = torch.tensor(df.fillna(0).values, dtype=torch.float32)

    # Initialize T-Patch-GNN model
    model = tPatchGNN()

    # Perform missing value imputation
    imputed_data = model.impute(data_tensor, missing_mask)

    # Convert back to DataFrame
    imputed_df = pd.DataFrame(imputed_data.numpy(), columns=df.columns)

    # Save to CSV
    imputed_df.to_csv(output_csv, index=False)
    print(f"Imputed file saved as: {output_csv}")

# Example usage
impute_missing_values("/Users/vedanshi/Documents/GitHub/practical-ml-project-2/main/data/test.csv", "output.csv")
