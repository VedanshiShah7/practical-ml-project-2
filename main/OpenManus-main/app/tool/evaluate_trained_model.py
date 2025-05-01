import os
import csv
import time
import torch
import datetime
import numpy as np
import pandas as pd

from pathlib import Path
from typing import Optional

from app.tool.base import BaseTool, ToolResult
from lib.parse_datasets import parse_datasets
from model.tPatchGNN import tPatchGNN
import lib.utils as utils


class EvaluateTrainedModel(BaseTool):
    name: str = "evaluate_trained_model"
    description: str = (
        "Loads a trained tPatchGNN model from a checkpoint and evaluates its performance on a dataset. "
        "Outputs evaluation metrics and optionally saves imputed values."
    )
    parameters: dict = {
        "type": "object",
        "properties": {
            "experiment_id": {
                "type": "string",
                "description": "ID of the trained experiment to load (e.g., 42 for 'experiment_42.ckpt')"
            },
            "save_path": {
                "type": "string",
                "description": "Directory path where the model checkpoints are stored.",
                "default": "experiments/"
            },
            "dataset": {
                "type": "string",
                "description": "Name of the dataset (e.g., physionet, mimic, ushcn).",
                "default": "physionet"
            }
        },
        "required": ["experiment_id"]
    }

    async def execute(self, **kwargs) -> ToolResult:
        experiment_id = kwargs["experiment_id"]
        save_path = kwargs.get("save_path", "experiments/")
        dataset = kwargs.get("dataset", "physionet")

        try:
            # Set defaults
            args = utils.dummy_args()
            args.load = experiment_id
            args.dataset = dataset
            args.save = save_path
            args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            args.batch_size = 1

            # Load checkpoint
            ckpt_path = os.path.join(args.save, f"experiment_{experiment_id}.ckpt")
            model, ckpt_args = self.load_ckpt(ckpt_path, device=args.device)

            # Dataset and evaluation
            data_obj = parse_datasets(ckpt_args, patch_ts=True)
            model.eval()
            with torch.no_grad():
                val_res = self.evaluation_eval(model, data_obj["val_dataloader"], data_obj["n_val_batches"])
                test_res = self.evaluation_eval(model, data_obj["test_dataloader"], data_obj["n_test_batches"])

            # Save evaluation metrics
            eval_file = f"evaluation_results_{experiment_id}.csv"
            with open(eval_file, mode='w', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(["Metric", "Validation", "Test"])
                for key in val_res:
                    writer.writerow([key, val_res[key], test_res[key]])

            # Save imputed values
            imputed_file = f"imputed_values_{experiment_id}.csv"
            with open(imputed_file, mode='w', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(["Time Point", "Observed Value", "Imputed Value", "Mask"])

                for _ in range(data_obj["n_test_batches"]):
                    batch_dict = utils.get_next_batch(data_obj["test_dataloader"])
                    if batch_dict is None:
                        continue

                    pred_y = model.forecasting(batch_dict["tp_to_predict"],
                                               batch_dict["observed_data"],
                                               batch_dict["observed_tp"],
                                               batch_dict["observed_mask"])
                    
                    observed = batch_dict["observed_data"].cpu().numpy()
                    predicted = pred_y.cpu().numpy()
                    mask = batch_dict["observed_mask"].cpu().numpy()

                    for i in range(len(observed)):
                        writer.writerow([i, observed[i], predicted[i], mask[i]])

            return ToolResult(
                output=f"✅ Evaluation complete.\nMetrics: {eval_file}\nImputed values: {imputed_file}"
            )

        except Exception as e:
            return ToolResult(error=f"❌ Failed to evaluate the model: {str(e)}")

    # In your tool definition file
    def EvaluateTrainedModel(experiment_id: str):
        import os
        import torch
        from tPatchGNN.experiments.experiment_runner import load_model, evaluate_model

        ckpt_path = f"/Users/vedanshi/Documents/GitHub/practical-ml-project-2/main/OpenManus-main/app/tool/tPatchGNN/tPatchGNN/experiments/experiment_{experiment_id}.ckpt"
        
        if not os.path.exists(ckpt_path):
            return f"Checkpoint file not found at {ckpt_path}. Please check the experiment ID."

        try:
            model = load_model(ckpt_path)  # you might need to adjust depending on your actual model code
            results = evaluate_model(model)  # expected to return a dict with metrics
            return f"✅ Model evaluated successfully.\nResults:\n{results}"
        except Exception as e:
            return f"❌ Failed to load/evaluate the model.\nError: {str(e)}"

    def load_ckpt(self, ckpt_path, device):
        if not os.path.exists(ckpt_path):
            raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
        checkpoint = torch.load(ckpt_path, map_location='cpu')
        ckpt_args = checkpoint["args"]
        model = tPatchGNN(ckpt_args)
        model.load_state_dict(checkpoint["state_dicts"])
        model.to(device)
        return model, ckpt_args

    def evaluation_eval(self, model, dataloader, n_batches):
        def compute_error(true, pred, mask, func, reduce):
            diff = (true - pred).abs() if func == "MAE" else (true - pred) ** 2
            masked = diff * mask
            if reduce == "sum":
                return masked.sum(), mask.sum()
            return masked.mean(), mask.sum()

        results = {"loss": 0, "mse": 0, "mae": 0, "rmse": 0, "mape": 0}
        n_eval = 0
        n_mape = 0

        for _ in range(n_batches):
            batch = utils.get_next_batch(dataloader)
            if batch is None:
                continue
            pred = model.forecasting(batch["tp_to_predict"],
                                     batch["observed_data"],
                                     batch["observed_tp"],
                                     batch["observed_mask"])

            mse, count = compute_error(batch["data_to_predict"], pred, batch["mask_predicted_data"], "MSE", "sum")
            mae, _ = compute_error(batch["data_to_predict"], pred, batch["mask_predicted_data"], "MAE", "sum")
            mape, mape_count = compute_error(batch["data_to_predict"], pred, batch["mask_predicted_data"], "MAPE", "sum")

            results["loss"] += mse
            results["mse"] += mse
            results["mae"] += mae
            results["mape"] += mape
            n_eval += count
            n_mape += mape_count

        count_nonzero = lambda x: torch.count_nonzero(x).item() if isinstance(x, torch.Tensor) else x

        for key in ["loss", "mse", "mae", "mape"]:
            results[key] = (results[key] / (n_eval + 1e-8)).item()
        results["rmse"] = np.sqrt(results["mse"])

        return results
