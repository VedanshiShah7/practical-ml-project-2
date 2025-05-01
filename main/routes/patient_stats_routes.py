from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
from scipy import stats

app = Flask(__name__)

# Load the dataset (Ensure your path is correct)
df = pd.read_csv("/Users/vedanshi/Documents/GitHub/practical-ml-project-2/main/data/test.csv")

def compute_patient_stats(df, icustayid):
    # Filter data for the specific icustayid
    patient_df = df[df['icustayid'] == icustayid]
    
    # Select numeric columns (exclude non-numeric columns such as icustayid)
    numeric_cols = patient_df.select_dtypes(include=[np.number]).columns.drop(['icustayid'], errors='ignore')
    
    # Aggregate statistics for the whole patient (across all rows for this icustayid)
    patient_stats = {}
    
    # Mean, std, min, max, variance
    patient_stats['mean'] = patient_df[numeric_cols].mean().to_dict()
    patient_stats['std'] = patient_df[numeric_cols].std().to_dict()
    patient_stats['min'] = patient_df[numeric_cols].min().to_dict()
    patient_stats['max'] = patient_df[numeric_cols].max().to_dict()
    patient_stats['variance'] = patient_df[numeric_cols].var().to_dict()  # sample variance
    
    # Z-scores (standardized values)
    z_scores = patient_df[numeric_cols].apply(lambda col: stats.zscore(col, nan_policy='omit'))
    patient_stats['z_scores'] = z_scores.mean().to_dict()  # Mean of z-scores for the patient
    
    # Percentile (rank)
    percentiles = patient_df[numeric_cols].rank(pct=True)
    patient_stats['percentiles'] = percentiles.mean().to_dict()  # Mean percentile for the patient
    
    # Interquartile Range (IQR) and Outliers based on IQR
    iqr = patient_df[numeric_cols].quantile(0.75) - patient_df[numeric_cols].quantile(0.25)
    lower = patient_df[numeric_cols].quantile(0.25) - 1.5 * iqr
    upper = patient_df[numeric_cols].quantile(0.75) + 1.5 * iqr
    outliers = ((patient_df[numeric_cols] < lower) | (patient_df[numeric_cols] > upper)).sum()  # Count outliers
    patient_stats['iqr_outliers'] = outliers.to_dict()
    
    # Trends: Using difference to calculate if there's an upward/downward trend
    trends = patient_df[numeric_cols].diff().apply(np.sign).fillna(0)  # 1 = increase, -1 = decrease, 0 = no change
    patient_stats['trends'] = trends.mean().to_dict()  # Mean trend across the patient
    
    # Return the patient statistics
    return patient_stats

@app.route('/patient_stats', methods=['GET'])
def get_patient_stats():
    # Get the icustayid from the query parameter
    icustayid = request.args.get('icustayid', type=int)
    
    if icustayid is None:
        return jsonify({'error': 'icustayid is required'}), 400
    
    # Compute patient stats for the given icustayid
    stats = compute_patient_stats(df, icustayid)
    
    # Return the stats as a JSON response
    return jsonify(stats)

if __name__ == '__main__':
    app.run(debug=True)
