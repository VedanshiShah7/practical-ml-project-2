import React, { useState, useEffect } from "react";
import axios from "../api/patientService";


function PatientProfile({ patient }) {
  const [data, setData] = useState(null);

  useEffect(() => {
    if (patient) {
      axios.get(`/${patient.pid}`).then((res) => setData(res.data));
    }
  }, [patient]);

  if (!data) return <p>Loading...</p>;

  return (
    <div>
      <h3>{data.name}'s Profile</h3>
      <ul>
        {Object.keys(data.imputed).map((key) => (
          <li key={key}>
            {key}: {data.imputed[key]}{" "}
            {data[key] === null ? <span>(Imputed)</span> : ""}
          </li>
        ))}
      </ul>
    </div>
  );
}

export default PatientProfile;
