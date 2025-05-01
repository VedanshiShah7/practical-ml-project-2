import pandas as pd
import numpy as np
from scipy import stats
from pydantic import BaseModel, Field
from typing import Optional
from app.tool.base import BaseTool, ToolResult

class CalculateStatisticsFromFile(BaseTool):
    name: str = "compute_patient_stats"
    description: str = (
        "Compute statistical metrics for a patient, given their icustayid. "
        "This includes mean, standard deviation, min, max, variance, "
        "z-scores, percentiles, IQR-based outliers, and trends over time."
    )
    parameters: dict = {
        "type": "object",
        "properties": {
            "data_path": {
                "type": "string",
                "description": "Path to the dataset file (CSV format).",
            },
            "icustayid": {
                "type": "integer",
                "description": "The icustayid for which patient statistics need to be computed.",
            },
        },
        "required": ["data_path", "icustayid"],
    }

    async def execute(self, **kwargs) -> ToolResult:
        data_path: Optional[str] = kwargs.get("data_path")
        icustayid: Optional[int] = kwargs.get("icustayid")
        output_file: Optional[str] = kwargs.get("output_file")  # Added for output file path

        if not data_path or not icustayid:
            return ToolResult(error="Both 'data_path' and 'icustayid' are required.")

        try:
            # Load the dataset
            df = pd.read_csv(data_path)
            
            # Filter data for the specific icustayid
            patient_df = df[df['icustayid'] == icustayid]
            
            # Select numeric columns (exclude non-numeric columns such as icustayid)
            numeric_cols = patient_df.select_dtypes(include=[np.number]).columns.drop(['icustayid'], errors='ignore')
            
            # Initialize the patient stats dictionary
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
            
            # Save the stats to a CSV file
            if output_file:
                patient_stats_df.to_csv(output_file, index=False)
                return ToolResult(output=f"Patient statistics computed and saved to {output_file}.", result=patient_stats_df)
            else:
                return ToolResult(output="Patient statistics computed successfully.", result=patient_stats_df)

        except Exception as e:
            return ToolResult(error=f"Error computing patient stats: {str(e)}")
