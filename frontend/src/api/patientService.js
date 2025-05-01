import axios from "axios";

export const fetchPatients = async () => {
  try {
    const response = await axios.get("/api/patients");
    return response.data;
  } catch (error) {
    console.error("Error fetching patients:", error);
    return [];
  }
};

export const fetchPatientById = async (patientId) => {
  try {
    const response = await axios.get(`/api/patient/${patientId}`);
    return response.data;
  } catch (error) {
    console.error("Error fetching patient:", error);
    return null;
  }
};
