import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import os
import google.generativeai as genai
from dotenv import load_dotenv
from test import chat_with_agent
import plotly.graph_objects as go

# Load API key from .env
load_dotenv()
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

# Load data
CSV_PATH = "/Users/vedanshi/Documents/GitHub/practical-ml-project-2/main/data/test.csv"

@st.cache_data
def load_data(path):
    return pd.read_csv(path)

df = load_data(CSV_PATH)

st.title("🩺 Patient Viewer + Chatbot Assistant")

# Patient selection
st.subheader("👤 Patient Lookup")
patient_ids = df["icustayid"].unique()
selected_id = st.selectbox("Select a patient icustayid", patient_ids)

patient_data = df[df["icustayid"] == selected_id]

if patient_data.empty:
    st.warning("No data found.")
    st.stop()

# Ensure proper datetime
if "charttime" in patient_data.columns:
    try:
        patient_data["charttime"] = pd.to_datetime(patient_data["charttime"])
        patient_data = patient_data.sort_values("charttime")
    except Exception as e:
        st.error("Could not convert charttime to datetime.")
        st.stop()
else:
    st.error("No charttime column found.")
    st.stop()

st.success(f"Showing data for patient {selected_id}")
st.dataframe(patient_data.head(20), use_container_width=True)

# Scrollable timeline
st.subheader("📈 Patient Timeline")
times = patient_data["charttime"].dt.strftime("%Y-%m-%d %H:%M:%S").tolist()
selected_index = st.slider("Select timepoint", 0, len(times)-1, len(times)//2)
selected_time = patient_data.iloc[selected_index]["charttime"]
st.markdown(f"Selected Time: **{selected_time}**")

# Show vital signs for selected time
numerical_cols = [col for col in patient_data.columns if patient_data[col].dtype in ['float64', 'int64']]
st.write("### Vital Stats at this Time")
st.table(patient_data.iloc[[selected_index]][numerical_cols].T.rename(columns={patient_data.index[selected_index]: "Value"}))

# Chart of vitals
st.write("### Trends Over Time (All Vitals - Interactive)")

fig = go.Figure()

for col in numerical_cols:
    fig.add_trace(go.Scatter(
        x=patient_data["charttime"],
        y=patient_data[col],
        mode='lines+markers',
        name=col
    ))

fig.update_layout(
    xaxis_title="Time",
    yaxis_title="Value",
    title="Vital Sign Trends Over Time",
    legend=dict(orientation="v", x=1.02, y=1),
    height=600,
    margin=dict(r=200)
)

st.plotly_chart(fig, use_container_width=True)


# Summary report with Gemini
st.subheader("🧾 Generate Patient Summary")

if st.button("Generate Report with Gemini"):
    st.info("Generating report...")
    latest_data = patient_data.iloc[selected_index][numerical_cols].dropna().to_dict()

    prompt = f"""
You are a medical assistant AI. Given the following vital signs at {selected_time} for patient {selected_id}, provide a brief summary of the patient's health:

{latest_data}

Summarize any anomalies and include possible suggestions or concerns in 100 words or less.
"""

    try:
        model = genai.GenerativeModel("gemini-pro")
        response = model.generate_content(prompt)
        st.success("✅ Summary Report")
        st.markdown(response.text)
    except Exception as e:
        st.error(f"Error generating summary: {e}")

# Chatbot
st.subheader("💬 Ask about this patient")

if 'chat_history' not in st.session_state:
    st.session_state['chat_history'] = []

user_input = st.text_input("Type your question here:")

if st.button("Send") and user_input:
    full_query = f"For patient with icustayid {selected_id}: {user_input}"
    with st.spinner("Thinking..."):
        try:
            response = chat_with_agent(full_query)
        except Exception as e:
            response = f"Error: {e}"
    st.session_state['chat_history'].append(("🧑‍⚕️ You", user_input))
    st.session_state['chat_history'].append(("🤖 Agent", response))

# Display chat history
st.markdown("---")
st.subheader("📜 Conversation History")
for speaker, message in st.session_state['chat_history']:
    bubble_style = '#DCF8C6' if speaker == "🧑‍⚕️ You" else '#E6ECF0'
    st.markdown(
        f"""
        <div style='background-color:{bubble_style}; color:black; padding:8px; border-radius:10px; margin-bottom:4px'>
            <b>{speaker}</b>: {message}
        </div>
        """,
        unsafe_allow_html=True
    )
