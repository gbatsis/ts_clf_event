import pandas as pd
from typing import Tuple

def split_data_time_based(data_path: str, test_size_percent: int, label_col: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Splits time-series data chronologically based on a specified percentage of rows for the test set.

    Args:
        data_path: The path to the CSV file containing the time series data.
        test_size_percent: The percentage of rows to include in the test set.
        label_col: The name of the column containing the class labels.

    Returns:
        train_df, test_df
    """

    df = pd.read_csv(data_path, index_col=0)

    test_size_rows = int(test_size_percent * len(df))

    if test_size_rows >= len(df):
        raise ValueError("test_size_rows must be smaller than the number of rows in the DataFrame.")

    df_sorted = df.sort_values('datetime')
    train_df = df_sorted.iloc[:-test_size_rows]
    test_df = df_sorted.iloc[-test_size_rows:]

    # Print the number of data points and class distribution in each set
    print("Number of data points in train set:", len(train_df))
    print("Number of data points in test set:", len(test_df))
    print("Class distribution in train set:", train_df[label_col].value_counts())
    print("Class distribution in test set:", test_df[label_col].value_counts())

    # Time in train and test set
    print("Time in train set:", train_df['datetime'].min(), "to", train_df['datetime'].max())
    print("Time in test set:", test_df['datetime'].min(), "to", test_df['datetime'].max())

    return train_df, test_df