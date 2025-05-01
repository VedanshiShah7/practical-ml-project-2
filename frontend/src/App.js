import React, { useState } from "react";
import PatientList from "./components/PatientList";
import PatientProfile from "./components/PatientProfile";
import Chatbot from "./components/ChatBot";
import "./App.css";

function App() {
  const [selectedPatient, setSelectedPatient] = useState(null);

  return (
    <div className="app-container">
      <div className="sidebar">
        <PatientList setSelectedPatient={setSelectedPatient} />
      </div>
      <div className="main-content">
        {selectedPatient ? (
          <PatientProfile patient={selectedPatient} />
        ) : (
          <h3>Select a patient to view details</h3>
        )}
        <Chatbot patient={selectedPatient} />
      </div>
    </div>
  );
}

export default App;
