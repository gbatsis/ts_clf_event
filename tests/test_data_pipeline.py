import pandas as pd
import numpy as np
import pytest
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler
from ts_clf_event.data_handler.data_preprocessing import (
    DataPreprocessor,
    FeatureEngineeringTransformer,
    DataFrameTransformer,
    split_data_time_based
)

# Sample data for testing
@pytest.fixture
def sample_data():
    data = {
        'datetime': pd.to_datetime(['2023-01-01', '2023-01-02', '2023-01-03', '2023-01-04',
                                    '2023-01-05', '2023-01-06', '2023-01-07', '2023-01-08']),
        'provider': ['A', 'A', 'A', 'A', 'B', 'B', 'B', 'B'],
        'value': [10, 12, 15, np.nan, 20, 22, 25, 23],
        'level': [1, 2, 3, 4, 3, 4, np.nan, 6],
        'frequency': [50, 51, 52, 53, 52, 51, 50, 53],
        'speed': [1, 2, 3, 4, 5, 6, 7, 8],
        'process': [0, 0, 1, 1, 0, 0, 1, 1]
    }
    return pd.DataFrame(data)

# Test DataPreprocessor class
def test_data_preprocessor_pipeline(sample_data):
    # Instantiate DataPreprocessor
    windows = [2, 3]
    features_to_roll = ['value', 'level']
    diff_lags = [1]
    features_to_diff = ['value', 'level']
    groupby_col = 'provider'

    preprocessor = DataPreprocessor(
        windows=windows,
        features_to_roll=features_to_roll,
        diff_lags=diff_lags,
        features_to_diff=features_to_diff,
        groupby_col=groupby_col
    )

    # Get the pipeline
    pipeline = preprocessor.get_pipeline()

    # Check if the pipeline is an instance of sklearn.pipeline.Pipeline
    assert isinstance(pipeline, Pipeline)

    # Check the steps in the pipeline
    assert len(pipeline.steps) == 3
    assert isinstance(pipeline.steps[0][1], FeatureEngineeringTransformer)
    assert isinstance(pipeline.steps[1][1], DataFrameTransformer)
    assert isinstance(pipeline.steps[1][1].transformer, SimpleImputer)
    assert isinstance(pipeline.steps[2][1], DataFrameTransformer)
    assert isinstance(pipeline.steps[2][1].transformer, MinMaxScaler)

def test_feature_engineering_transformer(sample_data):
    # Instantiate FeatureEngineeringTransformer
    windows = [2, 3]
    features_to_roll = ['value', 'level']
    diff_lags = [1]
    features_to_diff = ['value', 'level']
    groupby_col = 'provider'

    transformer = FeatureEngineeringTransformer(
        windows=windows,
        features_to_roll=features_to_roll,
        diff_lags=diff_lags,
        features_to_diff=features_to_diff,
        groupby_col=groupby_col
    )

    # Fit and transform the data
    transformed_data = transformer.fit_transform(sample_data)

    # Check if the output is a DataFrame
    assert isinstance(transformed_data, pd.DataFrame)

    # Check if the expected columns are present
    expected_columns = [
        'value_rolling_mean_2', 'value_rolling_std_2',
        'value_rolling_min_2', 'value_rolling_max_2',
        'level_rolling_mean_2', 'level_rolling_std_2',
        'level_rolling_min_2', 'level_rolling_max_2',
        'value_rolling_mean_3', 'value_rolling_std_3',
        'value_rolling_min_3', 'value_rolling_max_3',
        'level_rolling_mean_3', 'level_rolling_std_3',
        'level_rolling_min_3', 'level_rolling_max_3',
        'value_diff_1', 'level_diff_1'
    ]
    for col in expected_columns:
        assert col in transformed_data.columns

    # Check for correct calculation of rolling mean (example)
    assert transformed_data['value_rolling_mean_2'].iloc[1] == pytest.approx(11.0)
    assert transformed_data['level_rolling_mean_3'].iloc[2] == pytest.approx(2.0)

# Test the split_data_time_based function
def test_split_data_time_based(tmpdir):
    # Create a temporary CSV file for testing
    data = pd.DataFrame({
        'datetime': pd.to_datetime(['2023-01-01', '2023-01-02', '2023-01-03', '2023-01-04', '2023-01-05']),
        'process': [0, 1, 0, 1, 0]
    })
    file_path = tmpdir.join("test_data.csv")
    data.to_csv(file_path, index=False)

    # Test the function
    train_df, test_df = split_data_time_based(str(file_path), 0.4, 'process')

    # Check if the split is correct
    assert len(train_df) == 3
    assert len(test_df) == 2
    assert train_df['datetime'].max() < test_df['datetime'].min()