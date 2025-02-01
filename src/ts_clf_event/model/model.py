import os
import pandas as pd
import numpy as np
import pickle

from typing import List, Dict, Any
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
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
        params: Dict[str, Any] = None 
    ):
        self.windows = windows
        self.features_to_roll = features_to_roll
        self.diff_lags = diff_lags
        self.features_to_diff = features_to_diff
        self.groupby_col = groupby_col
        self.params = params

        # Initialize the preprocessing pipeline
        self.pipeline = DataPreprocessor(
            windows=self.windows,
            features_to_roll=self.features_to_roll,
            diff_lags=self.diff_lags,
            features_to_diff=self.features_to_diff,
            groupby_col=self.groupby_col
        ).get_pipeline()

        # Initialize the model
        self.model = self.get_model(params=params)

        # Append the model to the preprocessing pipeline
        self.pipeline.steps.append(("model", self.model))

    def get_tuned_model(self, X: pd.DataFrame, y: pd.Series, param_grid: Dict[str, List[Any]], n_splits: int = 5) -> RandomForestClassifier:
        """
        Performs hyperparameter tuning using GridSearchCV with TimeSeriesSplit.

        Args:
            X (pd.DataFrame): Full dataset features.
            y (pd.Series): Full dataset labels.
            param_grid (Dict[str, List[Any]]): Hyperparameter grid to search.
            n_splits (int, optional): Number of splits for cross-validation. Defaults to 5.

        Returns:
            RandomForestClassifier: The best model found by GridSearchCV.
        """

        scoring = {
            "precision": make_scorer(precision_score, average="weighted", zero_division=0),  
            "recall": make_scorer(recall_score, average="weighted", zero_division=0),       
            "f1": make_scorer(f1_score, average="weighted", zero_division=0),
            "precision_pos": make_scorer(precision_score, average="binary", zero_division=0),
            "recall_pos": make_scorer(recall_score, average="binary", zero_division=0),
            "f1_pos": make_scorer(f1_score, average="binary", zero_division=0),
        }

        tscv = TimeSeriesSplit(n_splits=n_splits)

        grid_search = GridSearchCV(
            estimator=self.pipeline,
            param_grid=param_grid,
            scoring=scoring,
            refit="f1_pos",
            cv=tscv,
            n_jobs=-1,
            verbose=5,
        )

        # Fit the model using a subset of the data for hyperparameter tuning
        grid_search.fit(X, y)

        print("Best parameters set found on development set:")
        print(grid_search.best_params_)
     
        # Return best model, selected hyperparameters and th cv results
        return grid_search.best_estimator_, grid_search.best_params_, grid_search.cv_results_

    def train(self, X_train: pd.DataFrame, y_train: pd.Series, tune_hyperparameters: bool = False, n_splits: int = 5) -> None:
        """
        Train the model on the training data, optionally with hyperparameter tuning.

        Args:
            X_train (pd.DataFrame): Training features.
            y_train (pd.Series): Training labels.
            tune_hyperparameters (bool, optional): Whether to perform hyperparameter tuning. Defaults to False.
            n_splits (int, optional): Number of splits for cross-validation during tuning. Defaults to 5.
        """
        
        if tune_hyperparameters:

            project_root = Path(__file__).parent.parent.parent.parent
            cv_results_path = os.path.join(project_root, "output", "models", "cv_results.pkl")

            if not os.path.isfile(cv_results_path):
                param_grid = self.get_param_grid()

                # Get the best model with tuned hyperparameters
                self.pipeline, self.params, self.cv_results = self.get_tuned_model(X_train, y_train, param_grid, n_splits)

                # Save the cross-validation results
                self.save_cv_results()

            else:
                # Load the cross-validation results
                with open(cv_results_path, "rb") as f:
                    cv_results = pickle.load(f)

                self.params = cv_results["params"]

                # Extract best parameters without the "model__" prefix
                best_params = {k.replace("model__", ""): v for k, v in cv_results["params"].items()}

                self.cv_results = cv_results["cv_results"]

                # Train a new model with the best parameters
                self.model = self.get_model(params=best_params)

                # Replace default model with the best model
                self.pipeline.steps[-1] = ("model", self.model)

        else:
            self.pipeline.fit(X_train, y_train)

    def save_cv_results(self) -> None:
        project_root = Path(__file__).parent.parent.parent.parent

        if not os.path.isdir(os.path.join(project_root, "output", "models")):
            os.makedirs(os.path.join(project_root, "output", "models"))
        
        cv_results_path = os.path.join(project_root, "output", "models", "cv_results.pkl")

        cv_results = {
            "params": self.params,
            "cv_results": self.cv_results
        }

        with open(cv_results_path, "wb") as f:
            pickle.dump(cv_results, f)

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

    @staticmethod
    def get_model(params: Dict[str, Any] = None) -> RandomForestClassifier:
        """
        Initializes and returns the model based on the provided parameters.

        Args:
            params (Dict[str, Any], optional): Model parameters. Defaults to None.

        Returns:
            RandomForestClassifier: The instantiated model.
        """

        print("Model parameters:", params)

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
            "model__n_estimators": [50, 100, 200],
            "model__max_depth": [4, 10, 20],
            "model__min_samples_split": [2, 5, 10],
            "model__min_samples_leaf": [1, 2, 4],
            "model__class_weight": ["balanced", None]
        }
    
    @staticmethod
    def process_cv_results(cv_results: Dict[str, List[Any]]) -> pd.DataFrame:
        """
        Processes and returns the cross-validation results.

        Args:
            cv_results (Dict[str, List[Any]]): Cross-validation results.

        Returns:
            pd.DataFrame: Processed cross-validation results.
        """
        
        # Clean the dictionary: Remove keys starts with "split" and with param_
        cv_results = {k: v for k, v in cv_results.items() if not k.startswith("split") and not k.startswith("param_")}
        # Convert the dictionary to a DataFrame
        df = pd.DataFrame(cv_results)

        # Sort by rank_test_f1_pos
        df = df.sort_values("rank_test_f1_pos", ascending=True).reset_index(drop=True)

        # Drop all columns starts with "rank"
        df = df.drop([col for col in df.columns if col.startswith("rank")], axis=1)

        return df

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

    model_pipeline = ModelPipeline(
        windows=windows,
        features_to_roll=features_to_roll,
        diff_lags=diff_lags,
        features_to_diff=features_to_roll,
        groupby_col="provider",
        params=None # We will use GridSearchCV to find them.
    )

    model_pipeline.train(
        x_train,
        y_train,
        tune_hyperparameters=True,
        n_splits=5,
    )

    print(model_pipeline.process_cv_results(model_pipeline.cv_results))

