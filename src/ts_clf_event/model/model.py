import os
import pandas as pd
import numpy as np
import pickle

from typing import List, Dict, Any
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import TimeSeriesSplit, cross_validate
from sklearn.metrics import make_scorer, precision_score, recall_score, f1_score

from ts_clf_event.data_handler.data_preprocessing import DataPreprocessor

class ModelPipeline:
    """
    A class to create a combined pipeline for data preprocessing and Random Forest classification.
    Supports time-series cross-validation.

    Args:
        windows (List[int]): List of window sizes for rolling window features.
        features_to_roll (List[str]): List of features to roll.
        diff_lags (List[int]): List of lags for difference features.
        features_to_diff (List[str]): List of features to diff.
        groupby_col (str, optional): Column to group by. Defaults to "provider".
    """
    def __init__(
        self, 
        windows: List[int], 
        features_to_roll: List[str], 
        diff_lags: List[int], 
        features_to_diff: List[str], 
        groupby_col: str = "provider", 
    ):
        self.windows = windows
        self.features_to_roll = features_to_roll
        self.diff_lags = diff_lags
        self.features_to_diff = features_to_diff
        self.groupby_col = groupby_col

        # Initialize the preprocessing pipeline
        self.pipeline = DataPreprocessor(
            windows=self.windows,
            features_to_roll=self.features_to_roll,
            diff_lags=self.diff_lags,
            features_to_diff=self.features_to_diff,
            groupby_col=self.groupby_col
        ).get_pipeline()

        # Initialize the model
        self.model = self.get_model()

        # Append the model to the preprocessing pipeline
        self.pipeline.steps.append(("model", self.model))

    @staticmethod
    def get_model(params: Dict[str, Any] = None) -> RandomForestClassifier:
        """
        Initializes and returns the model based on the provided parameters.

        Args:
            params (Dict[str, Any], optional): Model parameters. Defaults to None.

        Returns:
            RandomForestClassifier: The instantiated model.
        """
        return RandomForestClassifier(
            **params, random_state=42
        ) if params else RandomForestClassifier(random_state=42, class_weight="balanced")
        
    @staticmethod
    def get_param_grid() -> Dict[str, List[Any]]:
        """
        Returns the hyperparameter grid for the model.

        Returns:
            Dict[str, List[Any]]: Default hyperparameter grid.
        """
        return {
            "n_estimators": [50, 100, 200],
            "max_depth": [4, 10, 20],
            "min_samples_split": [2, 5, 10],
            "min_samples_leaf": [1, 2, 4],
        }

    def train(self, X_train: pd.DataFrame, y_train: pd.Series) -> None:
        """
        Train the model on the training data.

        Args:
            X_train (pd.DataFrame): Training features.
            y_train (pd.Series): Training labels.
        """
        self.pipeline.fit(X_train, y_train)

    def cross_validate(self, X: pd.DataFrame, y: pd.Series, n_splits: int = 5) -> dict:
        """
        Perform time-series cross-validation.

        Args:
            X (pd.DataFrame): Full dataset features.
            y (pd.Series): Full dataset labels.
            n_splits (int, optional): Number of splits for cross-validation. Defaults to 5.

        Returns:
            dict: A dictionary containing cross-validation scores.
        """

        scoring = {
            "precision": make_scorer(precision_score, average="weighted", zero_division=0),  
            "recall": make_scorer(recall_score, average="weighted", zero_division=0),       
            "f1": make_scorer(f1_score, average="weighted", zero_division=0),
            "precision_pos": make_scorer(precision_score, average="binary", zero_division=0),
            "recall_pos": make_scorer(recall_score, average="binary", zero_division=0),
            "f1_pos": make_scorer(f1_score, average="binary", zero_division=0),
        }

        # Perform cross-validation
        tscv = TimeSeriesSplit(n_splits=n_splits)

        # Print the number of positive and negative samples in each split
        for train_index, test_index in tscv.split(X):
            y_train = y.iloc[train_index]
            y_test = y.iloc[test_index]
            print("Train set:", y_train.value_counts())
            print("Test set:", y_test.value_counts())

        cv_results = cross_validate(
            self.pipeline, 
            X, 
            y, 
            cv=tscv, 
            scoring=scoring,
            n_jobs=-1,  
            verbose=5,  
            return_train_score=False,
        )

        return cv_results

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Make predictions using the trained model.

        Args:
            X (pd.DataFrame): Input features.

        Returns:
            np.ndarray: Predicted labels.
        """
        return self.pipeline.predict(X)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        Make probability predictions using the trained model.

        Args:
            X (pd.DataFrame): Input features.

        Returns:
            np.ndarray: Predicted probabilities.
        """
        return self.pipeline.predict_proba(X)
    
    def save_model(self, save_name: str) -> None:
        """
        Save the trained model to a file.

        Args:
            save_name (str): Name to save the model.
        """

        # Model path project_root/output/models/{save_name}.pkl
        project_root = Path(__file__).parent.parent.parent.parent

        if not os.path.isdir(os.path.join(project_root, "output", "models")):
            os.makedirs(os.path.join(project_root, "output", "models"))

        model_path = os.path.join(project_root, "output", "models", f"{save_name}.pkl")

        with open(model_path, "wb") as f:
            pickle.dump(self.pipeline, f)

    def load_model(self, model_name: str) -> None:
        """
        Load the trained model from a file.

        Args:
            model_name (str): Name of the model to load.
        """
        # Model path project_root/output/models/{model_name}.pkl
        project_root = Path(__file__).parent.parent.parent.parent

        if not os.path.isdir(os.path.join(project_root, "output", "models")):
            raise ValueError("Model directory does not exist.")
        
        model_path = os.path.join(project_root, "output", "models", f"{model_name}.pkl")

        if not os.path.isfile(model_path):
            raise ValueError("Model file does not exist.")

        with open(model_path, "rb") as f:
            self.pipeline = pickle.load(f)


# Example workflow
if __name__ == "__main__":

    from ts_clf_event.data_handler.utils import split_data_time_based

    data_path = "/Users/georgebatsis/Documents/Projects/ts_clf_event/data/test_dataframe.csv"
    test_size_percent = 0.2
    label_col = "process"

    train_df, test_df = split_data_time_based(data_path, test_size_percent, label_col)

    windows = "auto" #[60, 90, 120]
    features_to_roll = ["value", "level", "frequency", "speed"]
    diff_lags = [1, 2]
    
    x_train = train_df.drop("process", axis=1)
    y_train = train_df["process"]

    x_test = test_df.drop("process", axis=1)
    y_test = test_df["process"]

    model = ModelPipeline(
        windows=windows,
        features_to_roll=features_to_roll,
        diff_lags=diff_lags,
        features_to_diff=features_to_roll,
        groupby_col="provider",
    )

    model.cross_validate(x_train, y_train)