import pandas as pd
from typing import List

class FeatureEngineer:
    """
    A class for extracting features from time series data, including window-based features and difference features.

    Args:
        windows (List[int]): A list of window sizes (in terms of number of rows) to use for
            calculating rolling window statistics.
        features_to_roll (List[str]): A list of feature names (column names in the
            DataFrame) for which to calculate rolling window statistics.
        diff_lags (List[int]): A list of lag values to use when calculating
            differences.
        features_to_diff (List[str]): A list of feature names for which to calculate
            differences.
        groupby_col (str, optional): The name of the column to use for grouping the
            data before applying rolling window or difference calculations.
            Defaults to "provider".
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

    def engineer_window_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculates rolling window statistics for the specified features and window
        sizes.

        Args:
            df (pd.DataFrame): The input DataFrame containing the time series data.

        Returns:
            pd.DataFrame: A new DataFrame with the engineered window features added.
        """

        df_engineered = df.copy()
        
        for window in self.windows:
            for feature in self.features_to_roll:
                df_engineered[
                    f"{feature}_rolling_mean_{window}"
                ] = df_engineered.groupby(self.groupby_col)[feature].transform(
                    lambda x: x.rolling(window).mean()
                )
                df_engineered[
                    f"{feature}_rolling_std_{window}"
                ] = df_engineered.groupby(self.groupby_col)[feature].transform(
                    lambda x: x.rolling(window).std()
                )
                df_engineered[
                    f"{feature}_rolling_min_{window}"
                ] = df_engineered.groupby(self.groupby_col)[feature].transform(
                    lambda x: x.rolling(window).min()
                )
                df_engineered[
                    f"{feature}_rolling_max_{window}"
                ] = df_engineered.groupby(self.groupby_col)[feature].transform(
                    lambda x: x.rolling(window).max()
                )

        return df_engineered

    def engineer_difference_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculates difference features for the specified features and lag values.

        Args:
            df (pd.DataFrame): The input DataFrame containing the time series data.

        Returns:
            pd.DataFrame: A new DataFrame with the engineered difference features
                added. The new column names will follow the pattern:
                '{feature_name}_diff_{lag}'.
        """
        df_engineered = df.copy()

        for diff_lag in self.diff_lags:
            for feature in self.features_to_diff:
                df_engineered[
                    f"{feature}_diff_{diff_lag}"
                ] = df_engineered.groupby(self.groupby_col)[feature].transform(
                    lambda x: x.diff(diff_lag)
                )

        return df_engineered

    def engineer_all_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculates all features (window-based and difference-based) for the input DataFrame.

        Args:
            df (pd.DataFrame): The input DataFrame.

        Returns:
            pd.DataFrame: A new DataFrame with all engineered features.
        """
        if not isinstance(df.index, pd.DatetimeIndex):
            try:
                df = df.set_index('datetime')
            except KeyError:
                raise ValueError("DataFrame must have a DatetimeIndex or a 'datetime' column.")

        df_engineered = self.engineer_window_features(df)
        df_engineered = self.engineer_difference_features(df_engineered)
        return df_engineered