import os
from pathlib import Path
from typing import Optional, Dict, Any

import numpy as np
import torch
from pydantic import Field

from app.tool.base import BaseTool, ToolResult

from lib.parse_datasets import parse_datasets
from model.tPatchGNN import tPatchGNN
import lib.utils as utils


class ImputePatientTool(BaseTool):
    name: str = "impute_patient"
    description: str = (
        "Loads a trained tPatchGNN model and imputes missing values for a single patient "
        "from the test dataset. Returns both original (with NaNs) and imputed data."
    )
    parameters: dict = {
        "type": "object",
        "properties": {
            "index": {
                "type": "integer",
                "description": "Index of the patient in the test dataset.",
                "default": 0
            },
        },
        "required": [],
    }

    async def execute(self, **kwargs) -> ToolResult:
        # Specific checkpoint path
        ckpt_path = "/Users/vedanshi/Documents/GitHub/practical-ml-project-2/main/OpenManus-main/app/tool/tPatchGNN/tPatchGNN/experiments/experiment_48851.ckpt"
        
        if not os.path.exists(ckpt_path):
            return ToolResult(error=f"Checkpoint not found at {ckpt_path}")

        index: int = kwargs.get("index", 0)
        
        try:
            # Load checkpoint
            checkpoint = torch.load(ckpt_path, map_location="cpu")  # Force loading on CPU
            ckpt_args = checkpoint["args"]
            state_dict = checkpoint["state_dicts"]

            # Load model
            model = tPatchGNN(ckpt_args)
            model.load_state_dict({k: v for k, v in state_dict.items() if k in model.state_dict()})
            model.to("cpu")  # Ensure the model is loaded onto CPU
            model.eval()

            # Prepare dataset
            ckpt_args.batch_size = 1
            ckpt_args.device = "cpu"
            data_obj = parse_datasets(ckpt_args, patch_ts=True)
            test_loader = data_obj["test_dataloader"]

            # Get specified patient
            for i in range(index + 1):
                patient_dict = utils.get_next_batch(test_loader)

            if patient_dict is None:
                return ToolResult(error="Patient sample could not be retrieved.")

            # Move data to CPU (ensure it's on the correct device)
            for k in patient_dict:
                val = patient_dict[k]
                if isinstance(val, np.ndarray):
                    patient_dict[k] = torch.from_numpy(val).float().to("cpu")
                else:
                    patient_dict[k] = val.to("cpu")

            with torch.no_grad():
                pred_y = model.forecasting(
                    patient_dict["tp_to_predict"],
                    patient_dict["observed_data"],
                    patient_dict["observed_tp"],
                    patient_dict["observed_mask"]
                )

                observed_data = patient_dict["observed_data"]
                observed_mask = patient_dict["observed_mask"].bool()
                imputed = observed_data.clone()
                imputed[~observed_mask] = pred_y[~observed_mask]

            # Prepare result
            result = {
                "time_points": patient_dict["tp_to_predict"].cpu().numpy().tolist(),
                "original": np.where(
                    patient_dict["observed_mask"].cpu().numpy() == 1,
                    patient_dict["observed_data"].cpu().numpy(),
                    None
                ).tolist(),
                "imputed": imputed.cpu().numpy().tolist(),
            }

            return ToolResult(output=result)

        except Exception as e:
            return ToolResult(error=f"Failed to impute patient: {str(e)}")
