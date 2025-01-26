import pandas as pd
import pytest
from ts_clf_event.data_handler.feat_engineering import FeatureEngineer  

# Sample DataFrame fixture
@pytest.fixture
def sample_df():
    data = {
        'datetime': pd.to_datetime(['2023-01-01 00:00:00', '2023-01-01 00:01:00', '2023-01-01 00:02:00',
                                    '2023-01-01 00:03:00', '2023-01-01 00:04:00', '2023-01-01 00:05:00',
                                    '2023-01-01 00:06:00', '2023-01-01 00:07:00', '2023-01-01 00:08:00',
                                    '2023-01-01 00:09:00']),
        'provider': ['A', 'A', 'A', 'A', 'A', 'B', 'B', 'B', 'B', 'B'],
        'value': [10, 12, 15, 13, 16, 20, 22, 25, 23, 26],
        'level': [1, 2, 3, 2, 4, 3, 4, 5, 4, 6]
    }
    return pd.DataFrame(data)

def test_engineer_window_features(sample_df):
    windows = [2, 3]
    features_to_roll = ['value', 'level']
    diff_lags = []
    features_to_diff = []
    feature_engineer = FeatureEngineer(windows, features_to_roll, diff_lags, features_to_diff, groupby_col='provider')

    df_engineered = feature_engineer.engineer_window_features(sample_df.copy())

    # Check if the new columns are added
    assert 'value_rolling_mean_2' in df_engineered.columns
    assert 'value_rolling_std_3' in df_engineered.columns
    assert 'level_rolling_min_2' in df_engineered.columns
    assert 'level_rolling_max_3' in df_engineered.columns

    # Check the calculated values
    assert pd.isna(df_engineered['value_rolling_mean_2'][0])
    assert df_engineered['value_rolling_mean_2'][1] == 11

    assert pd.isna(df_engineered['level_rolling_std_3'][5])
    assert pd.isna(df_engineered['level_rolling_std_3'][6])
    # Corrected assertion: std of [3, 4, 5] is 1.0
    assert df_engineered['level_rolling_std_3'][7] == pytest.approx(1.0)

def test_engineer_difference_features(sample_df):
    diff_lags = [1, 2]
    features_to_diff = ['value', 'level']
    feature_engineer = FeatureEngineer([], [], diff_lags, features_to_diff, groupby_col='provider')

    df_engineered = feature_engineer.engineer_difference_features(sample_df.copy())

    # Check if the new columns are added
    assert 'value_diff_1' in df_engineered.columns
    assert 'level_diff_2' in df_engineered.columns

    # Check the calculated values
    assert pd.isna(df_engineered['value_diff_1'][0])
    assert df_engineered['value_diff_1'][1] == 2

    assert pd.isna(df_engineered['level_diff_2'][5])
    assert pd.isna(df_engineered['level_diff_2'][6])
    assert df_engineered['level_diff_2'][7] == 2

def test_engineer_all_features(sample_df):
    windows = [2]
    features_to_roll = ['value']
    diff_lags = [1]
    features_to_diff = ['level']
    feature_engineer = FeatureEngineer(windows, features_to_roll, diff_lags, features_to_diff, groupby_col='provider')

    df_engineered = feature_engineer.engineer_all_features(sample_df.copy())

    # Check if both window and difference features are present
    assert 'value_rolling_mean_2' in df_engineered.columns
    assert 'level_diff_1' in df_engineered.columns

    # Check values
    assert pd.isna(df_engineered['value_rolling_mean_2'][0])
    assert df_engineered['value_rolling_mean_2'][1] == 11
    assert pd.isna(df_engineered['level_diff_1'][0])
    assert df_engineered['level_diff_1'][1] == 1