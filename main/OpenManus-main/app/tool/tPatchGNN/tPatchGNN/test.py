import os
from typing import Union, Type
import pandas as pd
import torch
import joblib

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
    # Limit to the first few rows for better readability
    return f"Descriptive statistics:\n{stats.head(10).to_string()}"

statistics_tool = StructuredTool(
    name="compute_statistics",
    func=compute_statistics,
    description="Compute statistics (mean, std, etc.) for the dataset. Requires csv_file_path.",
    args_schema=StatsInput
)

# -------------------------
# Tool 4: Load model (imputation or mortality)
# -------------------------
class ModelLoadInput(BaseModel):
    ckpt_path: str = Field(description="Model checkpoint path (for imputation)")

def load_model(ckpt_path: str) -> str:
    print("Loading model from:", ckpt_path)
    model, args = Inspector.load_ckpt(ckpt_path, torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    args.batch_size = 1
    args.n = 100
    data_obj = parse_datasets(args, patch_ts=True)
    global MODEL, DATA_OBJ
    MODEL = model
    DATA_OBJ = data_obj
    return "Model loaded successfully."

load_model_tool = Tool(
    name="load_model",
    func=load_model,
    description="Load model for data imputation or forecasting. Requires ckpt_path."
)

# -------------------------
# Tool 5: Missing value imputation (infer)
# -------------------------
def infer():
    print("Running imputation...")
    model = MODEL
    dataloader = DATA_OBJ['train_dataloader']
    n_batches = DATA_OBJ['n_train_batches']

    for _ in range(n_batches):
        batch_dict = utils.get_next_batch(dataloader)
        if batch_dict is None:
            continue
        pred_y = model.forecasting(
            batch_dict["tp_to_predict"],
            batch_dict["observed_data"],
            batch_dict["observed_tp"],
            batch_dict["observed_mask"]
        )
        print(pred_y)
    return "Imputation completed."

infer_tool = Tool(
    name="infer",
    func=infer,
    description="Run missing value imputation using the loaded model."
)

# -------------------------
# Tool 6: Predict mortality
# -------------------------
class MortalityPredictionInput(BaseModel):
    csv_file_path: str = Field(description="Path to the input data CSV for mortality prediction")
    model_path: str = Field(description="Path to the trained mortality prediction model (.pkl)")

def predict_mortality(csv_file_path: str, model_path: str) -> str:
    df = pd.read_csv(csv_file_path)

    # Drop non-feature columns if needed
    feature_df = df.drop(columns=["bloc", "charttime"], errors="ignore")

    # Handle missing values
    feature_df = feature_df.fillna(feature_df.mean(numeric_only=True))

    model = joblib.load(model_path)
    predictions = model.predict(feature_df)

    return f"Mortality predictions: {predictions.tolist()}"

predict_mortality_tool = StructuredTool(
    name="predict_mortality",
    func=predict_mortality,
    description="Predict patient mortality using the trained model. Requires csv_file_path and model_path.",
    args_schema=MortalityPredictionInput
)

# -------------------------
# Tool 7: Save data to a file
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

# Memory setup
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

# -------------------------
# Chat function
# -------------------------
def chat_with_agent(query: str) -> str:
    return agent_with_memory.invoke({"input": query}, config={"configurable": {"session_id": "<foo>"}})

# -------------------------
# Example usage
# -------------------------
# print(chat_with_agent(r"Predict mortality using C:\path\to\your\file.csv and model C:\path\to\mortality_model.pkl"))
# Define the log file path
LOG_FILE_PATH = "chat_history.log"

# Function to save conversation to a log file
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
            # Save the conversation to the log file
            save_to_log(user_input, response)
        except Exception as e:
            error_message = f"Error: {e}"
            print(error_message)
            # Save the error to the log file
            save_to_log(user_input, error_message)
