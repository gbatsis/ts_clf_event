

import os
import time
import requests

from pathlib import Path

from ts_clf_event.data_handler.utils import split_data_time_based
from ts_clf_event.inference.inference import Inference
from ts_clf_event.model.evaluator import Evaluator

def mock_inference(
        data_path: str =os.path.join(Path(__file__).parent.parent.parent.parent, "data", "test_dataframe.csv"),
        history_points: int = 80
        ) -> None:
    
    """
    Mock inference for testing purposes.

    Args:
        data_path (str, optional): Path to the data. Defaults to os.path.join(Path(__file__).parent.parent.parent.parent, "data", "test_dataframe.csv").
        history_window (int, optional): The number of historical data points to keep. Defaults to 80.

    Returns:
        None
    """
    
    test_size_percent = 0.2
    label_col = "process"

    # Split the data and keep the training as history and the test as inference
    _, inference_df = split_data_time_based(data_path, test_size_percent, label_col)

    
    y_true_list = []
    y_pred_list = []

    for index, row in inference_df.iterrows():

        # Start time
        start_time = time.time()

        # Drop the process column
        data = row.drop("process").to_dict()
        response = requests.post("http://localhost:8000/predict/", json=data)
        y_pred_prob = response.json()["probability"]

        y_true_list.append(row["process"])
        y_pred_list.append(y_pred_prob)

        # End time
        end_time = time.time()

        # Calculate the elapsed time
        elapsed_time = end_time - start_time

        print("Predicted for row", index, ":", y_pred_prob, "in", elapsed_time, "seconds")

    evaluator = Evaluator("./output/test_results")
    evaluator.report_metrics(y_true_list, y_pred_list)

if __name__ == "__main__":
    mock_inference()
    