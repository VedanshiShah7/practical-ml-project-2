import React, { useState } from "react";
import { Button } from "./ui/button"; // Adjust path based on your folder structure
import { Card, CardContent } from "./ui/card"; // Adjust path based on your folder structure

export default function SepsisEHR() {
  const [inputText, setInputText] = useState("");
  const [predictions, setPredictions] = useState(null);
  const [shapImage, setShapImage] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  // Handle text input change
  const handleInputChange = (event) => {
    setInputText(event.target.value);
  };

  // Handle text submission and analysis
  const handleAnalyze = async () => {
    if (!inputText.trim()) return alert("Please enter a patient ID.");

    setLoading(true);
    setError(null);

    try {
      const response = await fetch("http://localhost:5000/predict", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ patient_id: inputText }),
      });

      const data = await response.json();

      if (data.error) {
        setError(data.error);
        setPredictions(null);
        setShapImage(null);
      } else {
        setPredictions(data.predictions);
        setShapImage(`http://localhost:5000/${data.shap_plot}`);
      }

      // Display imputation quality
      if (data.imputation_quality) {
        alert(`Imputation Quality:\nMAE: ${data.imputation_quality.MAE}\nMSE: ${data.imputation_quality.MSE}\nConfidence: ${data.imputation_quality.Confidence.toFixed(2)}`);
      }

    } catch (err) {
      setError("An error occurred while processing the data.");
    } finally {
      setLoading(false);
    }
};


  return (
    <div className="flex flex-col items-center p-6 bg-gray-100 min-h-screen">
      <div className="max-w-lg w-full bg-white shadow-lg rounded-lg p-6">
        <Card className="p-4 mb-4">
          <CardContent className="flex flex-col items-center">
            <textarea
              value={inputText}
              onChange={handleInputChange}
              placeholder="Enter patient data here..."
              className="border p-3 w-full mb-4 rounded-lg shadow-sm h-32"
            ></textarea>
            <Button
              className="mt-4 px-6 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition duration-200"
              onClick={handleAnalyze}
              disabled={loading}
            >
              {loading ? "Analyzing..." : "Analyze Data"}
            </Button>
          </CardContent>
        </Card>

        {error && <div className="mt-4 text-red-500 text-center">{error}</div>}

        {predictions && (
          <Card className="mt-6 p-4 w-full max-w-lg">
            <CardContent>
              <h2 className="text-xl font-bold text-center">Predictions</h2>
              <pre className="bg-gray-100 p-4 rounded-lg">{JSON.stringify(predictions, null, 2)}</pre>
            </CardContent>
          </Card>
        )}

        {shapImage && (
          <Card className="mt-6 p-4 w-full max-w-lg">
            <CardContent>
              <h2 className="text-xl font-bold text-center">SHAP Explanation</h2>
              <div className="flex justify-center">
                <img
                  src={shapImage}
                  alt="SHAP Explanation"
                  className="max-w-full h-auto rounded-lg shadow-lg"
                />
              </div>
            </CardContent>
          </Card>
        )}
      </div>
    </div>
  );
}
