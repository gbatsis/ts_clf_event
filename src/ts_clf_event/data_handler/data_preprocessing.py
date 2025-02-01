import numpy as np
import pandas as pd

from typing import Union, Tuple, List, Dict
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.impute import SimpleImputer

from ts_clf_event.data_handler.utils import split_data_time_based, analyze_sampling_rate
from ts_clf_event.data_handler.feat_engineering import FeatureEngineer


class DataFrameTransformer(BaseEstimator, TransformerMixin):
    """
    A wrapper transformer to ensure that the output of a transformer remains a pandas DataFrame.

    Args:
       transformer (Union[TransformerMixin, object]): The scikit-learn transformer to apply.
    """
    def __init__(self, transformer):
        self.transformer = transformer

    def fit(self, X: pd.DataFrame, y: pd.Series = None) -> "DataFrameTransformer":
        """Fits the transformer to the input data.

        Args:
            X (pd.DataFrame): The input DataFrame.
            y (pd.Series, optional): The target variable. Defaults to None.

        Returns:
            DataFrameTransformer: The fitted transformer.
        """
        self.transformer.fit(X, y)
        return self  # Return self to ensure pipeline compatibility

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Applies the wrapped transformer and ensures the output is a DataFrame.

        Args:
            X (pd.DataFrame): The input DataFrame.

        Returns:
            pd.DataFrame: The transformed DataFrame.
        """
        transformed = self.transformer.transform(X)
        # Ensure the result is a DataFrame with the same index and updated column names
        return pd.DataFrame(
            transformed, 
            index=X.index, 
            columns=self.get_feature_names_out(X)
        )
    
    def get_feature_names_out(self, X: pd.DataFrame) -> List[str]:
        """Get feature names for the DataFrame, falling back to original names if not available.

        Args:
            X (pd.DataFrame): The input DataFrame.

        Returns:
            List[str]: The feature names.
        """
        if hasattr(self.transformer, "get_feature_names_out"):
            return self.transformer.get_feature_names_out(input_features=X.columns)
        else:
            return X.columns.tolist()

class FeatureEngineeringTransformer(BaseEstimator, TransformerMixin):
    """
    A scikit-learn compatible transformer for applying the FeatureEngineer.

    Args:
        windows (Union[str, List[int]]): List of window sizes for rolling window features.
        features_to_roll (List[str]): List of features to roll.
        diff_lags (List[int]): List of lags for difference features.
        features_to_diff (List[str]): List of features to diff.
        groupby_col (str, optional): Column to group by. Defaults to "provider".
    """
    def __init__(
        self, 
        windows: Union[str, List[int]], 
        features_to_roll: List[str], 
        diff_lags: List[int], 
        features_to_diff: List[str], 
        groupby_col: str = "provider"
    ):
        self.windows = windows
        self.features_to_roll = features_to_roll
        self.diff_lags = diff_lags
        self.features_to_diff = features_to_diff
        self.groupby_col = groupby_col
        
        self.feature_engineer = FeatureEngineer(
            windows, features_to_roll, diff_lags, features_to_diff, groupby_col
        )
    
    def fit(self, X: pd.DataFrame, y: pd.Series = None) -> "FeatureEngineeringTransformer":
        """Fits the transformer to the input data (no-op in this case).

        Args:
            X (pd.DataFrame): The input DataFrame.
            y (pd.Series, optional): The target variable. Defaults to None.

        Returns:
            FeatureEngineeringTransformer: The fitted transformer.
        """
        # No fitting required for feature engineering
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Applies the feature engineering transformations.

        Args:
            X (pd.DataFrame): Input DataFrame.

        Returns:
            pd.DataFrame: Transformed DataFrame with engineered features.
        """
        if not isinstance(X, pd.DataFrame):
            raise ValueError("Input data must be a pandas DataFrame.")
        
        x_proccessed = self.feature_engineer.engineer_all_features(X)
        self.feature_list = x_proccessed.columns.tolist()
        return x_proccessed

class DataPreprocessor:
    """
    A class to create a scikit-learn pipeline for time series classification,
    starting with feature engineering and handling missing values.

    Args:
        windows (Union[str, List[int]]): List of window sizes for rolling window features.
        features_to_roll (List[str]): List of features to roll.
        diff_lags (List[int]): List of lags for difference features.
        features_to_diff (List[str]): List of features to diff.
        groupby_col (str, optional): Column to group by. Defaults to "provider".
        scaler (str, optional): Scaler to use. Defaults to "minmax".
    """
    def __init__(
        self, 
        windows: Union[str, List[int]], 
        features_to_roll: List[str], 
        diff_lags: List[int], 
        features_to_diff: List[str], 
        groupby_col: str = "provider",
        scaler: str = "minmax"
    ):
        self.windows = windows
        self.features_to_roll = features_to_roll
        self.diff_lags = diff_lags
        self.features_to_diff = features_to_diff
        self.groupby_col = groupby_col
        self.scaler = scaler

    def get_pipeline(self) -> Pipeline:
        """Creates a scikit-learn pipeline with feature engineering and missing value handling.
 
        Returns:
            Pipeline: An sklearn Pipeline object.
        """
        
        if self.scaler == "minmax":
            scaler = MinMaxScaler()
        elif self.scaler == "standard":
            scaler = StandardScaler()
        else:
            raise ValueError("Invalid scaler specified. Choose 'minmax' or 'standard'.")

        pipeline = Pipeline([
            ("feature_engineering", FeatureEngineeringTransformer(
                self.windows, 
                self.features_to_roll, 
                self.diff_lags, 
                self.features_to_diff, 
                self.groupby_col
            )),
            ("imputer", DataFrameTransformer(SimpleImputer(strategy="mean"))),
            ("scaler", DataFrameTransformer(scaler))
        ])
        return pipeline


if __name__ == "__main__":
    # Example workflow
    data_path = "/Users/georgebatsis/Documents/Projects/ts_clf_event/data/test_dataframe.csv"
    test_size_percent = 0.4
    label_col = "process"

    train_df, test_df = split_data_time_based(data_path, test_size_percent, label_col)

    windows = "auto"
    features_to_roll = ["value", "level", "frequency", "speed"]
    diff_lags = [1, 2]
    
    preprocessor = DataPreprocessor(
        features_to_roll=features_to_roll,
        features_to_diff=features_to_roll,
        windows=windows,
        diff_lags=diff_lags,
        groupby_col="provider",
    )

    x_train = train_df.drop("process", axis=1)
    y_train = train_df["process"]

    x_test = test_df.drop("process", axis=1)
    y_test = test_df["process"]

    pipeline = preprocessor.get_pipeline()
    pipeline.fit(x_train, y_train)

    x_train_transformed = pipeline.transform(x_train)
    x_test_transformed = pipeline.transform(x_test)

    # Print all columns of the transformed DataFrame
    for col in x_train_transformed.columns:
        print(col, x_train_transformed[col])


    