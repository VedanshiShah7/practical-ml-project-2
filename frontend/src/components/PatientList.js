import React, { useState, useEffect } from "react";
import axios from "../api/patientService"; 
import { fetchPatients } from "../api/patientService.js";
import { fetchPatients, fetchPatientById } from "../api/patientService";

function PatientList({ setSelectedPatient }) {
  const [patients, setPatients] = useState([]);
  const [query, setQuery] = useState("");

  useEffect(() => {
    axios.get(`/list?query=${query}`).then((res) => setPatients(res.data));
  }, [query]);

  return (
    <div>
      <input
        type="text"
        placeholder="Search patients..."
        onChange={(e) => setQuery(e.target.value)}
      />
      <ul>
        {patients.map((patient) => (
          <li key={patient.pid} onClick={() => setSelectedPatient(patient)}>
            {patient.name}
          </li>
        ))}
      </ul>
    </div>
  );
}

export default PatientList;
