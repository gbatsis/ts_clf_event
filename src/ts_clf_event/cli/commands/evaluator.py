import os
import typer
from pathlib import Path

from ts_clf_event.data_handler.utils import split_data_time_based
from ts_clf_event.model.model import ModelPipeline
from ts_clf_event.model.evaluator import Evaluator

DATA_PATH = os.path.join(Path(__file__).parent.parent.parent.parent.parent, "data", "test_dataframe.csv")

evaluator_cli: typer.Typer = typer.Typer()

@evaluator_cli.command()
def test():
    """
    Evaluates an existing model.
    """

    test_size_percent = 0.2
    label_col = "process"

    _, test_df = split_data_time_based(DATA_PATH, test_size_percent, label_col)
    windows = "auto" 
    features_to_roll = ["value", "level", "frequency", "speed"]
    diff_lags = [1, 2]

    model_pipeline = ModelPipeline(
        windows=windows,
        features_to_roll=features_to_roll,
        diff_lags=diff_lags,
        features_to_diff=features_to_roll,
        groupby_col="provider",
        params=None
    )

    model_pipeline.load_model("RF_model")
    x_test = test_df.drop("process", axis=1)
    y_test = test_df["process"]
    y_pred_prob = model_pipeline.predict_proba(x_test)

    Evaluator().report_metrics(y_test, y_pred_prob[:, 1], threshold=0.5)