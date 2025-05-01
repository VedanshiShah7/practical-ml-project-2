from flask import Blueprint, request, jsonify
import google.generativeai as genai

chat_bp = Blueprint('chat_bp', __name__)
genai.configure(api_key="YOUR_GEMINI_API_KEY")

@chat_bp.route("/ask", methods=["POST"])
def ask_question():
    data = request.json
    query = data["question"]
    response = genai.generate(
        model="gemini-pro",
        prompt=f"Patient data: {data['patient_data']}\n\nQuestion: {query}"
    )
    return jsonify({"response": response.text})
