import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import os
import google.generativeai as genai
from dotenv import load_dotenv
from test import chat_with_agent
import plotly.graph_objects as go
from io import StringIO
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image as RLImage
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors
import tempfile
import plotly.io as pio
import re

# Load API key
load_dotenv()
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

# Load data
CSV_PATH = "/Users/vedanshi/Documents/GitHub/practical-ml-project-2/data/test.csv"

@st.cache_data
def load_data(path):
    return pd.read_csv(path)

df = load_data(CSV_PATH)

st.title("🩺 Patient Viewer + Chatbot Assistant")

# Patient Selection
st.subheader("👤 Patient Lookup")
patient_ids = df["icustayid"].unique()
selected_id = st.selectbox("Select a patient icustayid", patient_ids)

patient_data = df[df["icustayid"] == selected_id]

if patient_data.empty:
    st.warning("No data found.")
    st.stop()

# DateTime Parsing
if "charttime" in patient_data.columns:
    try:
        patient_data.loc[:, "charttime"] = pd.to_datetime(patient_data["charttime"])
        patient_data = patient_data.sort_values("charttime")
    except Exception:
        st.error("Could not convert charttime to datetime.")
        st.stop()
else:
    st.error("No charttime column found.")
    st.stop()

st.success(f"Showing data for patient {selected_id}")
st.dataframe(patient_data.head(20), use_container_width=True)

# Timeline
st.subheader("📈 Patient Timeline")
times = patient_data["charttime"].dt.strftime("%Y-%m-%d %H:%M:%S").tolist()
selected_index = st.slider("Select timepoint", 0, len(times)-1, len(times)//2)
selected_time = patient_data.iloc[selected_index]["charttime"]
st.markdown(f"Selected Time: **{selected_time}**")

# Vital Stats
numerical_cols = [col for col in patient_data.columns if patient_data[col].dtype in ['float64', 'int64']]
st.write("### Vital Stats at this Time")
st.table(patient_data.iloc[[selected_index]][numerical_cols].T.rename(columns={patient_data.index[selected_index]: "Value"}))

# Trend Chart
st.write("### Trends Over Time (All Vitals - Interactive)")
fig = go.Figure()
for col in numerical_cols:
    fig.add_trace(go.Scatter(x=patient_data["charttime"], y=patient_data[col], mode='lines+markers', name=col))

fig.update_layout(
    xaxis_title="Time",
    yaxis_title="Value",
    title="Vital Sign Trends Over Time",
    legend=dict(orientation="v", x=1.02, y=1),
    height=600,
    margin=dict(r=200)
)
st.plotly_chart(fig, use_container_width=True)

# Markdown formatter
def format_ai_summary(text):
    text = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", text)
    text = re.sub(r"\*(.+?)\*", r"<i>\1</i>", text)
    text = text.replace('\n', '<br/>')
    return text

# PDF Generator
def generate_patient_pdf(summary_text, vitals_dict, fig, patient_data):
    buffer = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
    doc = SimpleDocTemplate(buffer.name, pagesize=letter)
    styles = getSampleStyleSheet()
    heading_style = ParagraphStyle('HeadingStyle', parent=styles['Heading2'], fontSize=14, spaceAfter=10)
    normal_style = ParagraphStyle('NormalStyle', parent=styles['Normal'], fontSize=11, spaceAfter=6)

    story = []
    story.append(Paragraph("🩺 Patient Summary Report", styles['Title']))
    story.append(Spacer(1, 12))
    story.append(Paragraph(f"<b>Patient ID:</b> {selected_id}", normal_style))
    story.append(Paragraph(f"<b>Selected Time:</b> {selected_time}", normal_style))
    story.append(Spacer(1, 12))

    story.append(Paragraph("AI-Generated Summary", heading_style))
    formatted_summary = format_ai_summary(summary_text)
    story.append(Paragraph(formatted_summary, normal_style))
    story.append(Spacer(1, 12))

    story.append(Paragraph("Vital Statistics", heading_style))
    stats_table_data = [["Vital", "Mean", "Median", "Min", "Max", "Std Dev"]]
    for vital in vitals_dict:
        vital_data = patient_data[vital].dropna()
        stats = [
            vital,
            round(vital_data.mean(), 2),
            round(vital_data.median(), 2),
            round(vital_data.min(), 2),
            round(vital_data.max(), 2),
            round(vital_data.std(), 2)
        ]
        stats_table_data.append(stats)

    stats_table = Table(stats_table_data, hAlign='LEFT')
    stats_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.darkblue),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.whitesmoke),
        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
        ("FONTSIZE", (0, 0), (-1, -1), 9),
        ("BOTTOMPADDING", (0, 0), (-1, 0), 6),
    ]))
    story.append(stats_table)
    story.append(Spacer(1, 12))

    image_path = os.path.join(tempfile.gettempdir(), "vitals_plot.png")
    pio.write_image(fig, image_path, format="png", width=800, height=400)
    story.append(Paragraph("Trend Chart", heading_style))
    story.append(RLImage(image_path, width=500, height=250))

    doc.build(story)
    return buffer.name

# Summary Generation
st.subheader("🧾 Generate Patient Summary")
if st.button("Generate Report with Gemini"):
    with st.spinner("Generating report..."):
        latest_data = patient_data.iloc[selected_index][numerical_cols].dropna().to_dict()

        prompt = f"""
        You are a medical assistant AI. Given the following vital signs at {selected_time} for patient {selected_id}, provide a detailed analysis of the patient's health:

        {latest_data}

        In your response, analyze the patient's current state, any anomalies or significant findings in the vital signs, possible causes for any abnormalities, and provide a prognosis or action plan.
        """
        
        try:
            model = genai.GenerativeModel(model_name="gemini-1.5-pro")
            response = model.generate_content(prompt)
            response_text = response.text
            st.success("✅ Summary Report")
            st.markdown(response_text)

            pdf_path = generate_patient_pdf(response_text, latest_data, fig, patient_data)
            with open(pdf_path, "rb") as f:
                st.download_button(
                    label="📄 Download Detailed PDF Report",
                    data=f.read(),
                    file_name=f"patient_{selected_id}_summary.pdf",
                    mime="application/pdf"
                )

        except Exception as e:
            st.error(f"Error generating summary: {e}")

# Chatbot Assistant
st.subheader("💬 Ask about this patient")

if 'chat_history' not in st.session_state:
    st.session_state['chat_history'] = []

user_input = st.text_input("Type your question here:")

def try_extract_and_save_csv(agent_response, icustayid):
    if "```" not in agent_response:
        return None
    try:
        raw_table = agent_response.split("```")[1].strip()
        df = pd.read_csv(StringIO(raw_table), sep=r"\s+")
        filename = f"patient_{icustayid}.csv"
        file_path = os.path.join("/mnt/data", filename)
        df.to_csv(file_path, index=False)
        return file_path
    except Exception as e:
        return f"⚠️ Failed to parse table: {e}"

if st.button("Send") and user_input:
    full_query = f"For patient with icustayid {selected_id}: {user_input}"
    with st.spinner("Thinking..."):
        try:
            agent_response = chat_with_agent(full_query)
            if isinstance(agent_response, dict) and 'output' in agent_response:
                response_text = agent_response['output']
            else:
                response_text = str(agent_response)
        except Exception as e:
            response_text = f"Error: {e}"

    st.session_state['chat_history'].append(("🧑‍⚕️ You", user_input))

    parsed_df = None
    if "```" in response_text:
        try:
            raw_table = response_text.split("```")[1].strip()
            parsed_df = pd.read_csv(StringIO(raw_table), sep=r"\s+")
        except Exception:
            parsed_df = None

    if parsed_df is not None:
        st.session_state['chat_history'].append(("🤖 Agent", "Here is the table:"))
        st.table(parsed_df)

        file_path = try_extract_and_save_csv(response_text, selected_id)
        if file_path and not file_path.startswith("⚠️"):
            with open(file_path, "rb") as f:
                st.download_button(
                    label="⬇️ Download Table CSV",
                    data=f.read(),
                    file_name=os.path.basename(file_path),
                    mime="text/csv"
                )
    else:
        st.session_state['chat_history'].append(("🤖 Agent", response_text))

# Chat History
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
