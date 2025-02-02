from typing import Union
from pydantic import BaseModel, field_validator

class InferenceInput(BaseModel):
    """
    Pydantic model for validating input data for inference.

    Attributes:
        start_value (float): The starting value of the time series.
        value (float): The current value of the time series.
        speed (float): The speed of the time series.
        level (float): The level of the time series.
        frequency (float): The frequency of the time series.
        status (int): The status of the time series.
        datetime (str): The datetime of the time series.
        provider (Union[str, int]): The provider of the time series.
    """

    start_value: float
    value: float
    speed: float
    level: float
    frequency: float
    status: float
    datetime: str    
    provider: Union[str, int]

    @field_validator("provider")
    def validate_provider(cls, v):
        """Validates that the provider is an integer."""
        try:
            return int(v)
        except ValueError:
            raise ValueError("Provider must be an integer")
        
class InferenceOutput(BaseModel):
    """
    Pydantic model for the prediction response.

    Attributes:
        probability: The predicted probability of the positive class.
    """
    probability: float