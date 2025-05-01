from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from app.tool.base import ToolResult
from predict_mortality import PredictMortality  # Import the tool you defined earlier

app = FastAPI()

class PredictionRequest(BaseModel):
    train_path: str
    test_path: str

@app.post("/predict_mortality")
async def predict_mortality(request: PredictionRequest):
    """
    API endpoint to trigger mortality prediction based on the given dataset paths.
    """
    try:
        # Initialize the tool with the request data
        tool = PredictMortality()
        
        # Execute the tool's pipeline using the provided paths
        result: ToolResult = await tool.execute(
            train_path=request.train_path,
            test_path=request.test_path,
        )
        
        # Return success or failure message
        if result.error:
            raise HTTPException(status_code=500, detail=result.error)
        
        return {"message": result.output}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal Server Error: {str(e)}")


# To run the FastAPI app, use `uvicorn` from the terminal like this:
# uvicorn mortality_prediction_api:app --reload
