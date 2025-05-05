
# AI Agents for Sepsis EHR Analysis

This project explores the integration of AI agents into Electronic Health Record (EHR) analysis for early prediction and explanation of Sepsis mortality risk. It combines deep learning techniques with a LangChain-based conversational agent to assist healthcare providers in data-driven decision-making.

## 🚀 Project Overview

Sepsis is a life-threatening condition that requires early detection for effective treatment. Our project proposes an AI-powered conversational system that:
- Loads and preprocesses patient EHR data.
- Predicts Sepsis mortality using both traditional ML and deep learning models.
- Provides interpretable predictions using SHAP-based explanations.
- Is deployed as an interactive web application with natural language interaction.

## 🧠 Key Components

- **Data Loader Tool**: Allows CSV upload and basic summary statistics.
- **Preprocessing Tool**: Handles imputation and filtering of missing values.
- **Prediction Tool**: Uses a Decision Tree and a Deep Neural Network (DNN) to predict mortality.
- **Explanation Tool**: Applies SHAP to explain predictions.
- **LangChain Agent**: Interfaces with all tools via natural language prompts.
- **Frontend (React App)**: Enables interactive user input and visualization.

## 📁 Directory Structure

```
project-root/
│
├── frontend/                # React-based web interface
├── main.py                 # LangChain agent and tool integration
├── model.py                # DNN model architecture
├── predict.py              # Prediction tool using ML/DNN
├── preprocessing.py        # Imputation and filtering logic
├── explain.py              # SHAP-based explanation module
├── sepsis.py               # Shared helper functions
└── README.md               # You're here!
```

## 🛠️ Setup & Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/VedanshiShah7/practical-ml-project-2.git
   cd sepsis-ehr-agent
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt

3. Start the streamlit application
    ```bash
    streamlit run app.py

## 👩‍⚕️ Usage

Interact with the AI agent using natural language questions such as:
- Run imputation model for patient with id 200003 from the file "/Users/vedanshi/Documents/GitHub/practical-ml-project-2/main/data/test.csv" using the preloaded imputation model from path "/Users/vedanshi/Documents/GitHub/practical-ml-project-2/main/OpenManus-main/app/tool/tPatchGNN/tPatchGNN/experiment_48851.ckpt"
- Give me the mortality prediction for patient with id 200003 in the file "/Users/vedanshi/Documents/GitHub/practical-ml-project-2/main/data/test.csv"

## 📚 References

- SHAP: https://github.com/slundberg/shap
- LangChain: https://github.com/hwchase17/langchain
- MIMIC-III EHR Dataset

## View full report [here](https://docs.google.com/document/d/1RtC93yPIw8yH2gbDPiGU0s51s2EClknmpqLDXZLwH9s/edit?usp=sharing)


## 👨‍💻 Authors
Vedanshi Shah
Brandeis University - COSI 149B (Spring 2025)
