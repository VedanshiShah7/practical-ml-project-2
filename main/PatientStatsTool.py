import pandas as pd
import numpy as np
from scipy import stats

def compute_patient_stats(df, icustayid):
    # Filter data for the specific icustayid
    patient_df = df[df['icustayid'] == icustayid]
    
    # Select numeric columns (exclude non-numeric columns such as icustayid)
    numeric_cols = patient_df.select_dtypes(include=[np.number]).columns.drop(['icustayid'], errors='ignore')
    
    # Aggregate statistics for the whole patient (across all rows for this icustayid)
    patient_stats = {}
    
    # Mean, std, min, max, variance
    patient_stats['mean'] = patient_df[numeric_cols].mean()
    patient_stats['std'] = patient_df[numeric_cols].std()
    patient_stats['min'] = patient_df[numeric_cols].min()
    patient_stats['max'] = patient_df[numeric_cols].max()
    patient_stats['variance'] = patient_df[numeric_cols].var()  # sample variance
    
    # Z-scores (standardized values)
    z_scores = patient_df[numeric_cols].apply(lambda col: stats.zscore(col, nan_policy='omit'))
    patient_stats['z_scores'] = z_scores.mean()  # Mean of z-scores for the patient
    
    # Percentile (rank)
    percentiles = patient_df[numeric_cols].rank(pct=True)
    patient_stats['percentiles'] = percentiles.mean()  # Mean percentile for the patient
    
    # Interquartile Range (IQR) and Outliers based on IQR
    iqr = patient_df[numeric_cols].quantile(0.75) - patient_df[numeric_cols].quantile(0.25)
    lower = patient_df[numeric_cols].quantile(0.25) - 1.5 * iqr
    upper = patient_df[numeric_cols].quantile(0.75) + 1.5 * iqr
    outliers = ((patient_df[numeric_cols] < lower) | (patient_df[numeric_cols] > upper)).sum()  # Count outliers
    patient_stats['iqr_outliers'] = outliers
    
    # Trends: Using difference to calculate if there's an upward/downward trend
    trends = patient_df[numeric_cols].diff().apply(np.sign).fillna(0)  # 1 = increase, -1 = decrease, 0 = no change
    patient_stats['trends'] = trends.mean()  # Mean trend across the patient
    
    # Combine all stats into a DataFrame for easier viewing
    patient_stats_df = pd.DataFrame(patient_stats)
    
    # Return the patient statistics
    return patient_stats_df

# Example usage
if __name__ == "__main__":
    # Load dataset (replace with your path)
    df = pd.read_csv("/Users/vedanshi/Documents/GitHub/practical-ml-project-2/main/data/test.csv")
    
    # Specify the icustayid you're interested in
    icustayid = 200003  # Example icustayid
    
    # Get patient statistics for the given icustayid
    patient_stats = compute_patient_stats(df, icustayid)
    
    # Output the statistics
    print(patient_stats)

