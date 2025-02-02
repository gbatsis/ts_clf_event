import os
import typer
from pathlib import Path

from ts_clf_event.data_handler.utils import split_data_time_based
from ts_clf_event.model.model import ModelPipeline

DATA_PATH = os.path.join(Path(__file__).parent.parent.parent.parent.parent, "data", "test_dataframe.csv")

trainer_cli: typer.Typer = typer.Typer()

@trainer_cli.command()
def train_hyper():
    """
    Trains a model after hyperparameter tuning.
    """
    test_size_percent = 0.2
    label_col = "process"

    train_df, _ = split_data_time_based(DATA_PATH, test_size_percent, label_col)
    windows = "auto" 
    features_to_roll = ["value", "level", "frequency", "speed"]
    diff_lags = [1, 2]
    
    x_train = train_df.drop("process", axis=1)
    y_train = train_df["process"]

    model_pipeline = ModelPipeline(
        windows=windows,
        features_to_roll=features_to_roll,
        diff_lags=diff_lags,
        features_to_diff=features_to_roll,
        groupby_col="provider",
        params=None
    )

    model_pipeline.train(
        x_train,
        y_train,
        tune_hyperparameters=True,
        n_splits=5,
    )

    print(model_pipeline.process_cv_results(model_pipeline.cv_results))

    # Re-train the model with the best hyperparameters
    model_pipeline.train(
        x_train,
        y_train,
        tune_hyperparameters=False,
    )

    model_pipeline.save_model("RF_model")