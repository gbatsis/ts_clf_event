import warnings
import pandas as pd
from typing import List, Dict

from ts_clf_event.data_handler.utils import analyze_sampling_rate
from ts_clf_event.utils.logging import setup_logger

logger = setup_logger()

# Ignore warnings
warnings.filterwarnings("ignore")

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
        sampling_conf: str = "mode",
    ):
        self.windows = windows
        self.features_to_roll = features_to_roll
        self.diff_lags = diff_lags
        self.features_to_diff = features_to_diff
        self.groupby_col = groupby_col
        self.sampling_conf = sampling_conf

    def determine_windows(self, df: pd.DataFrame) -> Dict[str, List[int]]:
        """
        Determines window sizes based on the sampling rate analysis for each provider.

        Args:
            df (pd.DataFrame): The input DataFrame.

        Returns:
            Dict[str, List[int]]: A dictionary where keys are provider IDs and values are lists of window sizes.
        """
        
        def round_to_nearest(numbers: list, base: int) -> list:
            """Rounds each number in a list to the nearest multiple of a base value.

            Args:
                numbers: The list of numbers to round.
                base: The base value (e.g., 10, 30).

            Returns:
                A new list with the rounded numbers.
            """
            return [base * round(num / base) for num in numbers]
        
        sampling_rate_info = analyze_sampling_rate(df, self.groupby_col)
        windows = {}
        for provider in sampling_rate_info:
            mode = sampling_rate_info[provider][self.sampling_conf]
            windows[provider] = [mode * multiplier for multiplier in [1, 1.5, 2]]
        
        # Round the windows to the nearest 10 or 30
        for provider in windows:
            windows[provider] = round_to_nearest(windows[provider], 10)

        # Logging the windows for each provider
        for provider in windows:
            logger.info(f"Windows for {provider}: {windows[provider]}")
        
        return windows

    def engineer_window_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculates rolling window statistics for the specified features and window
        sizes, using provider-specific windows if applicable.

        Args:
            df (pd.DataFrame): The input DataFrame containing the time series data.

        Returns:
            pd.DataFrame: A new DataFrame with the engineered window features added.
        """

        df_engineered = df.copy()

        if self.windows == "auto":
            self.windows = self.determine_windows(df)

        for provider in df[self.groupby_col].unique():
            for feature in self.features_to_roll:
                if isinstance(self.windows, dict):
                    # Use provider-specific windows
                    provider_windows = self.windows[int(provider)]
                else:
                    # Use the same windows for all providers
                    provider_windows = self.windows

                for window in provider_windows:
                    df_engineered.loc[df[self.groupby_col] == provider, f"{feature}_rolling_mean_{window}"] = \
                        df[df[self.groupby_col] == provider][feature].rolling(window).mean()
                    df_engineered.loc[df[self.groupby_col] == provider, f"{feature}_rolling_std_{window}"] = \
                        df[df[self.groupby_col] == provider][feature].rolling(window).std()
                    df_engineered.loc[df[self.groupby_col] == provider, f"{feature}_rolling_min_{window}"] = \
                        df[df[self.groupby_col] == provider][feature].rolling(window).min()
                    df_engineered.loc[df[self.groupby_col] == provider, f"{feature}_rolling_max_{window}"] = \
                        df[df[self.groupby_col] == provider][feature].rolling(window).max()

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
                # Set index and drop the existed Unnamed: 0 column
                df = df.set_index("datetime")
                if "Unnamed: 0" in df.columns:
                    df = df.drop("Unnamed: 0", axis=1)
            except KeyError:
                raise ValueError("DataFrame must have a DatetimeIndex or a 'datetime' column.")

        df_engineered = self.engineer_window_features(df)
        df_engineered = self.engineer_difference_features(df_engineered)
        
        # Feature list
        self.feature_list = df_engineered.columns.tolist()
        
        return df_engineered