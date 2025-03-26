import React, { useState } from "react";
import { Button } from "@/components/ui/button";
import { Card, CardContent } from "@/components/ui/card";
import { Upload } from "lucide-react";

export default function SepsisEHR() {
  const [file, setFile] = useState(null);
  const [predictions, setPredictions] = useState(null);
  const [shapImage, setShapImage] = useState(null);

  const handleFileChange = (event) => {
    setFile(event.target.files[0]);
  };

  const handleUpload = async () => {
    if (!file) return alert("Please select a file first.");
    const formData = new FormData();
    formData.append("file", file);

    const response = await fetch("https://your-backend-url/upload", {
      method: "POST",
      body: formData,
    });
    const data = await response.json();
    setPredictions(data.predictions);
    setShapImage(data.shap_plot);
  };

  return (
    <div className="flex flex-col items-center p-6">
      <Card className="p-4 w-full max-w-lg">
        <CardContent className="flex flex-col items-center">
          <input type="file" accept=".csv" onChange={handleFileChange} />
          <Button className="mt-4" onClick={handleUpload}>
            <Upload className="mr-2" /> Upload & Analyze
          </Button>
        </CardContent>
      </Card>

      {predictions && (
        <Card className="mt-6 p-4 w-full max-w-lg">
          <CardContent>
            <h2 className="text-xl font-bold">Predictions</h2>
            <pre>{JSON.stringify(predictions, null, 2)}</pre>
          </CardContent>
        </Card>
      )}

      {shapImage && (
        <Card className="mt-6 p-4 w-full max-w-lg">
          <CardContent>
            <h2 className="text-xl font-bold">SHAP Explanation</h2>
            <img src={`https://your-backend-url/${shapImage}`} alt="SHAP Explanation" />
          </CardContent>
        </Card>
      )}
    </div>
  );
}
