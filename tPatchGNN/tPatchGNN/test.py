import time
import os
from typing import Union, Type
import pandas as pd
import torch
import joblib
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
import tensorflow as tf
from sklearn.metrics import accuracy_score

from pydantic import BaseModel, Field
from langchain.tools import Tool, StructuredTool
from langchain.prompts import ChatPromptTemplate
from langchain.schema import BaseMessage, SystemMessage
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory

from dotenv import load_dotenv
from run_samples import Inspector
from lib.parse_datasets import parse_datasets
import lib.utils as utils

# Load .env variables
load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")
print("Using Gemini API key:", api_key[:5] + "*****")
os.environ["GOOGLE_API_KEY"] = api_key

# Global variables
MODEL, DATA_OBJ = None, None
DF = None

# -------------------------
# Tool 1: Load and filter CSV by icustayid
# -------------------------
class CSVLoadInput(BaseModel):
    csv_file_path: str = Field(description="Path to CSV file")
    ids: Union[int, str] = Field(description="icustayid of the patient")

def load_csv(csv_file_path: str, ids: Union[int, str]) -> str:
    df = pd.read_csv(csv_file_path)
    filtered_df = df[df['icustayid'] == float(ids)]
    if filtered_df.empty:
        return f"No data found for icustayid: {ids}"
    global DF
    DF = filtered_df
    return filtered_df.to_string(index=False)

load_csv_tool = StructuredTool(
    name="load_csv",
    func=load_csv,
    description="Load CSV and filter by 'icustayid' to retrieve a patient's data. Requires csv_file_path and ids.",
    args_schema=CSVLoadInput
)

# -------------------------
# Tool 2: Filter by icustayid and charttime
# -------------------------
class PatientDateFilterInput(BaseModel):
    csv_file_path: str = Field(description="Path to CSV file")
    patient_id: Union[int, str] = Field(description="icustayid of the patient")
    charttime: Union[int, float] = Field(description="Encounter charttime (e.g., 7245486000.0)")

def filter_by_patient_and_date(csv_file_path: str, patient_id: Union[int, str], charttime: Union[int, float]) -> str:
    df = pd.read_csv(csv_file_path)
    filtered = df[(df["icustayid"] == float(patient_id)) & (df["charttime"] == float(charttime))]
    if filtered.empty:
        return "No data found for the given icustayid and charttime."
    return filtered.to_string(index=False)

patient_date_tool = StructuredTool(
    name="filter_by_patient_and_date",
    func=filter_by_patient_and_date,
    description="Filter data by icustayid and charttime. Requires csv_file_path, patient_id, and charttime.",
    args_schema=PatientDateFilterInput
)

# -------------------------
# Tool 3: Compute statistics
# -------------------------
class StatsInput(BaseModel):
    csv_file_path: str = Field(description="Path to CSV file")

def compute_statistics(csv_file_path: str) -> str:
    df = pd.read_csv(csv_file_path)
    stats = df.describe(include='all').transpose()
    return f"Descriptive statistics:\n{stats.head(10).to_string()}"

statistics_tool = StructuredTool(
    name="compute_statistics",
    func=compute_statistics,
    description="Compute statistics (mean, std, etc.) for the dataset. Requires csv_file_path.",
    args_schema=StatsInput
)

# -------------------------
# Tool 4: Load model (imputation)
# -------------------------
class ModelLoadInput(BaseModel):
    ckpt_path: str = Field(description="Model checkpoint path (for imputation)")

def load_model(ckpt_path: str) -> str:
    print("Loading model from:", ckpt_path)
    import argparse
    import torch
    import sys
    import os

    # Add the path so the module is discoverable
    sys.path.append("/app/tool")

    torch.serialization.add_safe_globals([argparse.Namespace])

    try:
        loaded = torch.load(
            ckpt_path,
            map_location=torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )
    except Exception as e:
        return f"❌ Failed to load checkpoint: {e}"

    if isinstance(loaded, dict) and "state_dicts" in loaded and "args" in loaded:
        print("Loading from state_dicts format...")
        args = loaded["args"]
        state_dict = loaded["state_dicts"]
        import sys
        sys.path.append('app/tool')

        from tPatchGNN.model.tPatchGNN import tPatchGNN

        model = tPatchGNN(args).to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        model.load_state_dict(state_dict)
    else:
        return "❌ Unexpected checkpoint format."

    try:
        data_obj = parse_datasets(args, patch_ts=True)
    except Exception as e:
        return f"✅ Model loaded, but dataset parsing failed: {e}"

    global MODEL, DATA_OBJ
    MODEL = model
    DATA_OBJ = data_obj
    return "✅ Model and dataset loaded successfully."

load_model_tool = Tool(
    name="load_model",
    func=load_model,
    description="Load model for data imputation or forecasting. Requires ckpt_path.",
    args_schema=ModelLoadInput
)

# -------------------------
# Tool 5: Run imputation
# -------------------------
def infer(_=None):
    print("Running imputation...")

    model = MODEL
    dataloader = DATA_OBJ['train_dataloader']
    n_batches = min(DATA_OBJ['n_train_batches'], 10)

    # Ensure directory exists
    output_dir = "imputed_batches"
    os.makedirs(output_dir, exist_ok=True)

    preview = None
    successful_batches = 0

    for i in range(n_batches):
        print(f"\n🔄 Processing batch {i+1}/{n_batches}")
        batch_dict = utils.get_next_batch(dataloader)
        if batch_dict is None:
            print("⚠️ Batch is None — skipping.")
            continue

        start = time.time()
        try:
            pred_y = model.forecasting(
                batch_dict["tp_to_predict"],
                batch_dict["observed_data"],
                batch_dict["observed_tp"],
                batch_dict["observed_mask"]
            )

            pred_np = pred_y.detach().cpu().numpy()
            batch_shape = pred_np.shape

            # Flatten prediction for CSV saving: (B, T, F) -> (B*T, F)
            pred_flat = pred_np.reshape(-1, pred_np.shape[-1])

            df = pd.DataFrame(pred_flat)
            file_path = os.path.join(output_dir, f"imputed_batch_{i+1}.csv")
            df.to_csv(file_path, index=False)

            if preview is None:
                preview = df.head().to_string(index=False)

            print(f"✅ Saved to {file_path}")
            successful_batches += 1
            print(f"⏱️ Forecasting time: {time.time() - start:.2f} seconds")

        except Exception as e:
            print("❌ Error during forecasting:", e)
            continue

    if successful_batches == 0:
        return "❌ No batches were successfully imputed."

    return (
        f"✅ Imputation completed for {successful_batches} batch(es).\n"
        f"📁 Output saved in the folder: `{output_dir}`.\n"
        f"📊 Preview of first batch:\n{preview}"
    )

infer_tool = Tool(
    name="infer",
    func=infer,
    description="Run missing value imputation using the loaded model. No parameters required."
)

# -------------------------
# Tool 6: Predict mortality (Updated with path and improved output)
# -------------------------
class MortalityPredictionInput(BaseModel):
    icustayid: Union[int, str] = Field(description="ICU stay ID of the patient to predict mortality for")

def predict_mortality(icustayid: Union[int, str]) -> str:
    csv_file_path = "/Users/vedanshi/Documents/GitHub/practical-ml-project-2/tPatchGNN/tPatchGNN/predictions_dnn.csv"
    
    try:
        df = pd.read_csv(csv_file_path, usecols=["icustayid", "Probability of Survival (0)", "Probability of Mortality (1)"])
    except Exception as e:
        return f"❌ Error reading CSV: {e}"

    # Normalize both icustayid and df values to strings without '.0'
    def normalize(val):
        val_str = str(val)
        return val_str.rstrip('.0') if val_str.endswith('.0') else val_str

    target_id = normalize(icustayid)
    df["icustayid_normalized"] = df["icustayid"].apply(normalize)

    matched_rows = df[df["icustayid_normalized"] == target_id]

    if matched_rows.empty:
        return f"❌ icustayid '{icustayid}' not found in prediction file."

    # Take the average probability across all rows for that icustayid
    avg_prob = matched_rows["Probability of Mortality (1)"].mean()

    # Return risk message based on threshold
    if avg_prob > 50.0:
        return (
            f"⚠️ High risk of mortality for icustayid {icustayid}.\n"
            f"🧮 Average Probability of Mortality: {avg_prob:.2f}%\n"
            f"🔎 Please consider early intervention."
        )
    else:
        return (
            f"✅ Low risk of mortality for icustayid {icustayid}.\n"
            f"🧮 Average Probability of Mortality: {avg_prob:.2f}%\n"
            f"📋 Continue regular monitoring."
        )
    
predict_mortality_tool = StructuredTool(
    name="predict_mortality",
    func=predict_mortality,
    description="Predict mortality using a fixed CSV file and row index (icustayid).",
    args_schema=MortalityPredictionInput,
    return_direct=True  # 👈 ensures raw output is returned to user
)

# -------------------------
# Tool 7: Save data
# -------------------------
class SaveDataInput(BaseModel):
    data: str = Field(description="The data to save (as a string).")
    file_path: str = Field(description="The file path where the data should be saved.")

def save_data(data: str, file_path: str) -> str:
    try:
        with open(file_path, "w") as file:
            file.write(data)
        return f"Data successfully saved to {file_path}"
    except Exception as e:
        return f"Failed to save data: {e}"

save_data_tool = StructuredTool(
    name="save_data",
    func=save_data,
    description="Save data to a specified file. Requires the data (as a string) and the file path.",
    args_schema=SaveDataInput
)

# -------------------------
# LangChain setup
# -------------------------
tools = [
    load_csv_tool,
    patient_date_tool,
    statistics_tool,
    load_model_tool,
    infer_tool,
    predict_mortality_tool,
    save_data_tool
]

llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash",
    temperature=0.2
)

prompt = ChatPromptTemplate.from_messages([
    SystemMessage(content="""You are a helpful assistant capable of general conversation and medical data analysis. 
                             Use tools when working with CSV data, models, or predictions.
                             Always display the results from the tools clearly in your response.
"""),
    ("placeholder", "{chat_history}"),
    ("human", "{input}"),
    ("placeholder", "{agent_scratchpad}")
])

agent = create_tool_calling_agent(llm, tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

class InMemoryHistory(BaseChatMessageHistory, BaseModel):
    messages: list[BaseMessage] = Field(default_factory=list)

    def add_messages(self, messages: list[BaseMessage]) -> None:
        self.messages.extend(messages)

    def clear(self) -> None:
        self.messages = []

memory = InMemoryHistory()

agent_with_memory = RunnableWithMessageHistory(
    agent_executor,
    lambda session_id: memory,
    input_messages_key="input",
    history_messages_key="chat_history"
)

def chat_with_agent(query: str) -> str:
    return agent_with_memory.invoke({"input": query}, config={"configurable": {"session_id": "<foo>"}})

LOG_FILE_PATH = "chat_history.log"

def save_to_log(user_input: str, agent_response: str) -> None:
    with open(LOG_FILE_PATH, "a") as log_file:
        log_file.write(f"You: {user_input}\n")
        log_file.write(f"Agent: {agent_response}\n")
        log_file.write("-" * 50 + "\n")

if __name__ == "__main__":
    print("Welcome to the chat interface! Type 'exit' to quit.")
    while True:
        user_input = input("You: ")
        if user_input.lower() == "exit":
            print("Goodbye!")
            break
        try:
            response = chat_with_agent(user_input)
            print(f"Agent: {response}")
            save_to_log(user_input, response)
        except Exception as e:
            error_message = f"Error: {e}"
            print(error_message)
            save_to_log(user_input, error_message)
