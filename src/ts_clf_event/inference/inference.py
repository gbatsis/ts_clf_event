import pandas as pd

from typing import Union, Dict, List

from ts_clf_event.model.model import ModelPipeline

class Inference:
    """
    Class for making predictions using a trained model.

    Args:
        history_df (pd.DataFrame): The historical data.
        windows (List[int]): List of window sizes for rolling window features.
        features_to_roll (List[str]): List of features to roll.
        diff_lags (List[int]): List of lags for difference features.
        features_to_diff (List[str]): List of features to diff.
        groupby_col (str, optional): Column to group by. Defaults to "provider".
        history_window (int, optional): The number of historical data points to keep. Defaults to 80.
        model_name (str, optional): The name of the model. Defaults to "RF_model".
    """
    def __init__(
        self, 
        history_df: pd.DataFrame,
        windows: Union[str, List[int]],
        features_to_roll: List[str], 
        diff_lags: List[int], 
        features_to_diff: List[str], 
        groupby_col: str = "provider",
        history_window: int = 80,
        model_name: str = "RF_model" 
        ):

        self.windows = windows
        self.features_to_roll = features_to_roll
        self.diff_lags = diff_lags
        self.features_to_diff = features_to_diff
        self.groupby_col = groupby_col
        self.history_df = history_df
        self.history_window = history_window
        self.model_name = model_name

        self.model = self.model_init()
        
    def model_init(self):
        """
        Initializes the model pipeline.

        Returns:
            ModelPipeline (object): The initialized model pipeline.
        """
        model = ModelPipeline(
            windows=self.windows,
            features_to_roll=self.features_to_roll,
            diff_lags=self.diff_lags,
            features_to_diff=self.features_to_roll,
            groupby_col=self.groupby_col,
        )

        # Load the model
        model.load_model(self.model_name)

        print("Model pipeline:")
        print(model.pipeline)

        return model

    def predict(self, data: Dict) -> None:
        """
        Predicts the probability for a single data point, utilizing historical data for feature engineering.

        Args:
            data (Dict): New data point features.

        Returns:
            float: Predicted probability.
        """

        self.check_data(data)

        # Convert the dictionary to a DataFrame
        new_data_df = pd.DataFrame([data])

        # Ensure the new data has a datetime column for time-based operations
        if "datetime" not in new_data_df.columns:
            new_data_df["datetime"] = pd.to_datetime("now")

        # Append new data to historical data for feature engineering
        combined_df = pd.concat([self.history_df, new_data_df], ignore_index=True)

        y_pred = self.model.predict_proba(combined_df)

        self.history_df = pd.concat([self.history_df, new_data_df], ignore_index=True)

        self.history_df = self.history_df.groupby(self.groupby_col).tail(self.history_window)
        self.history_df = self.history_df.reset_index(drop=True).sort_values("datetime")
        
        return y_pred[-1, 1]

    def check_data(self, data: Dict) -> None:
        """
        Checks if the input data for inference are consistent.
        Expects a dictionary with the following structure:

        {
            start_value:,
            value:,
            speed:,
            level:,
            frequency:,
            status:,
            provider:
        }

        Args:
            data (Dict): New data point features.

        Raises:
            ValueError: If the required columns are missing.
        """

        required_columns = ["start_value", "value", "speed", "level", "frequency", "status", "provider"]
        for col in required_columns:
            if col not in data:
                raise ValueError(f"Missing required column: {col}")

        