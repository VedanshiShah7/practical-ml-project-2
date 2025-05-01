from flask import Flask
from routes.patient_routes import patient_bp
from routes.chat_routes import chat_bp
import os
import sys

# Add tPatchGNN to sys.path
tpatch_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../t-PatchGNN/tPatchGNN"))
sys.path.insert(0, tpatch_path)

print(f"Added tPatchGNN to PYTHONPATH: {tpatch_path}")


app = Flask(__name__)
app.register_blueprint(patient_bp, url_prefix="/api/patient")
app.register_blueprint(chat_bp, url_prefix="/api/chat")

if __name__ == "__main__":
    app.run(debug=True)
