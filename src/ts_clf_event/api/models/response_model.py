from pydantic import BaseModel

class PredictionResponse(BaseModel):
    """
    Pydantic model for the prediction response.

    Attributes:
        probability: The predicted probability of the positive class.
    """
    probability: float