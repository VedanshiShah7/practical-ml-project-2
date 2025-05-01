import React, { useState, useEffect } from "react";

export default function PatientDashboard() {
  const [patients, setPatients] = useState([]);
  const [searchQuery, setSearchQuery] = useState("");
  const [selectedPatient, setSelectedPatient] = useState(null);
  const [imputationData, setImputationData] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  // Fetch patient list from the backend
  useEffect(() => {
    fetch("http://localhost:5000/patients")
      .then((response) => response.json())
      .then((data) => setPatients(data))
      .catch((err) => {
        console.error("Error fetching patients:", err);
        setError("Failed to fetch patient data.");
      });
  }, []);

  // Handle search input change
  const handleSearch = (event) => {
    setSearchQuery(event.target.value);
  };

  // Fetch data for the selected patient
  const handleSelectPatient = async (patientId) => {
    setLoading(true);
    setError(null);
    setSelectedPatient(patientId);
    try {
      const response = await fetch(`http://localhost:5000/predict/${patientId}`);
      const data = await response.json();
      setImputationData(data);
    } catch (err) {
      setError("Failed to fetch imputation data.");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="flex flex-col items-center p-6 bg-gray-100 min-h-screen">
      {!selectedPatient ? (
        <div className="max-w-lg w-full bg-white shadow-lg rounded-lg p-6">
          <input
            type="text"
            placeholder="Search patient ID..."
            value={searchQuery}
            onChange={handleSearch}
            className="mb-4 w-full border border-gray-300 p-3 rounded-lg"
          />
          <ul className="w-full">
            {patients
              .filter((patient) =>
                patient.toLowerCase().includes(searchQuery.toLowerCase())
              )
              .map((patient) => (
                <li key={patient} className="mb-2">
                  <button
                    className="w-full bg-blue-500 text-white p-3 rounded-lg"
                    onClick={() => handleSelectPatient(patient)}
                  >
                    {patient}
                  </button>
                </li>
              ))}
          </ul>
        </div>
      ) : (
        <div className="max-w-2xl w-full bg-white shadow-lg rounded-lg p-6">
          <button
            onClick={() => setSelectedPatient(null)}
            className="mb-4 bg-gray-500 text-white p-3 rounded-lg"
          >
            Back to Patient List
          </button>
          {loading ? (
            <p>Loading...</p>
          ) : error ? (
            <p className="text-red-500">{error}</p>
          ) : imputationData ? (
            <>
              <h2 className="text-xl font-bold">Patient {selectedPatient} - Imputation Data</h2>
              <pre className="bg-gray-100 p-4 rounded-lg mt-4">
                {JSON.stringify(imputationData.predictions, null, 2)}
              </pre>
              {imputationData.imputation_quality && (
                <div className="mt-4">
                  <h3 className="text-lg font-bold">Imputation Quality</h3>
                  <p>MAE: {imputationData.imputation_quality.MAE.toFixed(4)}</p>
                  <p>MSE: {imputationData.imputation_quality.MSE.toFixed(4)}</p>
                  <p>Confidence: {imputationData.imputation_quality.Confidence.toFixed(2)}</p>
                </div>
              )}
              {imputationData.shap_plot && (
                <div className="mt-6">
                  <h3 className="text-lg font-bold">SHAP Explanation</h3>
                  <img
                    src={`http://localhost:5000/${imputationData.shap_plot}`}
                    alt="SHAP Explanation"
                    className="max-w-full h-auto rounded-lg shadow-lg"
                  />
                </div>
              )}
            </>
          ) : null}
        </div>
      )}
    </div>
  );
}
