
from pathlib import Path
from fastapi import FastAPI, HTTPException, Request

from ts_clf_event.api.utils.data_utils import extract_history
from ts_clf_event.inference.inference import Inference
from ts_clf_event.api.models import InputModel, PredictionResponse

DATA_PATH = Path(__file__).parent.parent.parent.parent / "data" / "test_dataframe.csv"
HISTORY_POINTS = 80
WINDOWS = "auto"
FEATURES_TO_ROLL = ["value", "level", "frequency", "speed"]
DIFF_LAGS = [1, 2]
GROUP_BY_COL = "provider"

# Initialize FastAPI app
app = FastAPI(
    title="Time Series Classification Inference API",
    description="API for making predictions with a trained time-series classification model.",
    version="0.1.0",
)

history_df = extract_history(DATA_PATH, HISTORY_POINTS)
inference = Inference(history_df, WINDOWS, FEATURES_TO_ROLL, DIFF_LAGS, FEATURES_TO_ROLL, GROUP_BY_COL, HISTORY_POINTS)

@app.post("/predict/", response_model=PredictionResponse)
async def predict(data: InputModel, request: Request):
    """
    Predicts the probability of the positive class for a new data point.

    Args:
        data (InputData): The input data point.
        request (Request): The request object, used to get the client's IP address for demonstration.

    Returns:
        PredictionResponse: The predicted probability.
    """
    #try:
        # Process the sample data

    # Pydantic to dict
    sample = data.model_dump()

    # Get the predicted probability
    y_pred_prob = inference.predict(sample)

    # Get the client's host from the request headers.
    client_host = request.client.host

    # Log the prediction.
    print(f"Prediction for client {client_host}: {y_pred_prob:.4f}")

    return {"probability": y_pred_prob}

    #except Exception as e:
    #    raise HTTPException(status_code=500, detail=str(e))



#uvicorn ts_clf_event.api.app:app --reload
