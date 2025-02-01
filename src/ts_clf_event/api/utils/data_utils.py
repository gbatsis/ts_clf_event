import pandas as pd

from ts_clf_event.data_handler.utils import split_data_time_based

def extract_history(data_path, history_points, groupby_col="provider", label_col="process") -> pd.DataFrame:
    """
    Extracts historical data for feature engineering.

    Args:
        data_path (str): Path to the CSV file containing data.
        history_points (int): Number of recent rows to keep per provider.
        groupby_col (str): Column to group by (e.g., 'provider').
        label_col (str): Label column to drop.

    Returns:
        pd.DataFrame: DataFrame containing historical data.
    """
    history_df, _ = split_data_time_based(data_path, test_size_percent=0.2, label_col=label_col)
    history_df = history_df.groupby(groupby_col).tail(history_points)
    history_df = history_df.drop(label_col, axis=1)

    return history_df


