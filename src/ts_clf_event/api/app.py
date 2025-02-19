from pathlib import Path
from fastapi import FastAPI, HTTPException, Request

from ts_clf_event.api.utils import extract_history
from ts_clf_event.inference.inference import Inference
from ts_clf_event.api.models import InferenceInput, InferenceOutput
from ts_clf_event.utils.logging import setup_logger

logger = setup_logger()

DATA_PATH = Path(__file__).parent.parent.parent.parent / "data" / "test_dataframe.csv"
HISTORY_POINTS = 1000
WINDOWS = "auto"
FEATURES_TO_ROLL = ["value", "level", "frequency", "speed"]
DIFF_LAGS = [1, 2]
GROUP_BY_COL = "provider"

app = FastAPI(
    title="Time Series Event Classification API",
    description="API for making predictions with a trained time series classification model.",
    version="0.1.0"
)

# Log loading
logger.info(f"Loading historical data from: {DATA_PATH}")

history_df = extract_history(DATA_PATH, HISTORY_POINTS)
inference = Inference(history_df, WINDOWS, FEATURES_TO_ROLL, DIFF_LAGS, FEATURES_TO_ROLL, GROUP_BY_COL, HISTORY_POINTS)

@app.post("/predict/", response_model=InferenceOutput)
async def predict(data: InferenceInput, request: Request):
    """
    Predicts the probability of the positive class for a new data point.

    Args:
        data (InputData): The input data point.
        request (Request): The request object, used to get the client's IP address for demonstration.

    Returns:
        PredictionResponse: The predicted probability.
    """
    try:
    
        # Pydantic to dict
        sample = data.model_dump()

        # Get the predicted probability
        y_pred_prob = inference.predict(sample)

        # Get the client's host from the request headers.
        client_host = request.client.host

        # Log the prediction.
        logger.info(f"Predicted probability for {client_host}: {y_pred_prob}")

        return {"probability": y_pred_prob}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Healthcheck
@app.get("/health")
async def root():
    return {"message": "Healthy"}


# Command to run the API
#uvicorn ts_clf_event.api.app:app --reload