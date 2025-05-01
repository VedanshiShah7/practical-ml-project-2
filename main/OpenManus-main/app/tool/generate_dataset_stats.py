import pandas as pd
import numpy as np
from scipy import stats
from pathlib import Path
from typing import Optional

from app.tool.base import BaseTool, ToolResult


class GenerateDatasetStats(BaseTool):
    name: str = "generate_dataset_stats"
    description: str = (
        "Generate comprehensive statistics from a tabular dataset (CSV). "
        "Includes row-wise, column-wise, rolling, delta, and trend statistics. "
        "Can return a full stats CSV and optionally return a summary for a specific icustayid (patient ID)."
    )
    parameters: dict = {
        "type": "object",
        "properties": {
            "filepath": {
                "type": "string",
                "description": "Path to the input CSV file containing the dataset.",
            },
            "output_path": {
                "type": "string",
                "description": "Optional path to save the resulting stats CSV. "
                               "If not provided, saves as 'dataset_stats_only.csv' in the same directory.",
            },
            "patient_id": {
                "type": "number",
                "description": "Optional: icustayid of a specific patient to extract and summarize statistics.",
            },
        },
        "required": ["filepath"],
    }

    async def execute(self, **kwargs) -> ToolResult:
        filepath: str = kwargs.get("filepath")
        output_path: Optional[str] = kwargs.get("output_path")
        patient_id: Optional[float] = kwargs.get("patient_id")

        try:
            df = pd.read_csv(filepath)
            numeric_cols = df.select_dtypes(include=[np.number]).columns.drop(
                ['bloc', 'icustayid', 'charttime'], errors='ignore'
            )

            # Row-wise statistics
            row_stats = pd.DataFrame(index=df.index)
            row_stats['row_mean'] = df[numeric_cols].mean(axis=1)
            row_stats['row_std'] = df[numeric_cols].std(axis=1)
            row_stats['row_min'] = df[numeric_cols].min(axis=1)
            row_stats['row_max'] = df[numeric_cols].max(axis=1)
            row_stats['row_sum'] = df[numeric_cols].sum(axis=1)

            # Column-wise stats
            z_scores = df[numeric_cols].apply(lambda col: stats.zscore(col, nan_policy='omit'))
            percentiles = df[numeric_cols].rank(pct=True)
            iqr = df[numeric_cols].quantile(0.75) - df[numeric_cols].quantile(0.25)
            lower = df[numeric_cols].quantile(0.25) - 1.5 * iqr
            upper = df[numeric_cols].quantile(0.75) + 1.5 * iqr
            iqr_outliers = ((df[numeric_cols] < lower) | (df[numeric_cols] > upper)).astype(int)
            iqr_outliers.columns = [f'{col}_iqr_outlier' for col in iqr_outliers.columns]

            # Rolling stats
            rolling_mean = df[numeric_cols].rolling(window=3, min_periods=1).mean()
            rolling_mean.columns = [f'{col}_rollmean' for col in rolling_mean.columns]
            rolling_std = df[numeric_cols].rolling(window=3, min_periods=1).std()
            rolling_std.columns = [f'{col}_rollstd' for col in rolling_std.columns]

            # Deltas and trends
            deltas = df[numeric_cols].diff().fillna(0)
            deltas.columns = [f'{col}_delta' for col in deltas.columns]
            trends = df[numeric_cols].diff().apply(np.sign).fillna(0)
            trends.columns = [f'{col}_trend' for col in trends.columns]

            # Combine all
            stats_df = pd.concat([
                df,
                row_stats,
                z_scores.add_suffix('_zscore'),
                percentiles.add_suffix('_percentile'),
                iqr_outliers,
                rolling_mean,
                rolling_std,
                deltas,
                trends
            ], axis=1)

            output_file = Path(output_path) if output_path else Path(filepath).parent / "dataset_stats_only.csv"
            stats_df.to_csv(output_file, index=False)

            output_message = f"✅ Stats CSV saved at: {output_file.resolve()}"

            # Handle individual patient statistics if patient_id is provided
            if patient_id is not None:
                patient_rows = stats_df[stats_df['icustayid'] == patient_id]
                if patient_rows.empty:
                    return ToolResult(
                        output=output_message,
                        error=f"No rows found for patient id {patient_id}."
                    )

                # Create a per-patient summary from the stats
                patient_summary = patient_rows.describe(include='all').transpose().round(3).to_dict()

                return ToolResult(
                    output=output_message + f"\n📊 Summary statistics for patient {patient_id}:",
                    data=patient_summary
                )

            return ToolResult(output=output_message)

        except Exception as e:
            return ToolResult(error=f"❌ Failed to generate dataset stats: {str(e)}")
