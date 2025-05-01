import pandas as pd
import re
from pathlib import Path
from typing import Optional, Dict, Any

from app.tool.base import BaseTool


class RetrievePatientValues(BaseTool):
    name: str = "retrieve_patient_values"
    description: str = (
        "Retrieve rows for a patient using their ICU stay ID (icustayid) from a CSV file. "
        "Understands requests like 'Get data for patient 321' or 'Fetch rows for ID 102'."
    )

    parameters: Dict[str, Any] = {
        "type": "object",
        "properties": {
            "request": {
                "type": "string",
                "description": "Natural language query to extract ICU stay ID (e.g. 'Get data for patient 321')",
            },
            "filepath": {
                "type": "string",
                "description": "Path to the CSV file containing patient data with an 'icustayid' column.",
            },
            "output_path": {
                "type": "string",
                "description": "Optional path to save the output CSV file with matched patient rows.",
            },
        },
        "required": ["request", "filepath"],
    }

    def _extract_icustayid(self, request: str) -> Optional[int]:
        """Extract icustayid from natural language request."""
        match = re.search(r"(?:patient\s*id|patient|id)\s*(\d+)", request, re.IGNORECASE)
        return int(match.group(1)) if match else None

    async def execute(
        self,
        request: str,
        filepath: str,
        output_path: Optional[str] = None,
    ) -> Dict[str, Any]:
        try:
            df = pd.read_csv(filepath)
            icustayid = self._extract_icustayid(request)

            if icustayid is None:
                return {
                    "observation": "❌ Could not extract an ICU stay ID (icustayid) from your request.",
                    "success": False,
                }

            patient_rows = df[df['icustayid'] == icustayid]

            if patient_rows.empty:
                return {
                    "observation": f"⚠️ No data found for ICU stay ID {icustayid}.",
                    "success": False,
                }

            base_dir = Path(output_path) if output_path else Path(filepath).parent
            observed_file = base_dir / "observed_patients.csv"
            patient_rows.to_csv(observed_file, index=False)

            return {
                "observation": f"✅ Found and saved rows for ICU stay ID {icustayid} to {observed_file.resolve()}",
                "success": True,
                "data": {
                    "num_rows": len(patient_rows),
                    "file": str(observed_file.resolve()),
                },
            }

        except Exception as e:
            return {
                "observation": f"❌ Failed to process request: {str(e)}",
                "success": False,
            }
