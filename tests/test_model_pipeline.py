import pytest
import pandas as pd
import numpy as np
from sklearn.utils.validation import check_is_fitted
from ts_clf_event.model.model import ModelPipeline


@pytest.fixture
def mock_data():
    """Fixture to create a mock dataset for testing."""
    data = {
        "datetime": pd.date_range("2023-01-01", periods=100, freq="T"),
        "value": np.random.rand(100),
        "level": np.random.randint(0, 10, size=100),
        "frequency": np.random.rand(100),
        "speed": np.random.rand(100),
        "provider": np.random.choice([1, 2], size=100),
        "process": np.random.choice([0, 1], size=100),
    }
    df = pd.DataFrame(data, index=data["datetime"])
    x = df.drop(columns=["process", "datetime"])
    y = df["process"]
    return x, y

@pytest.fixture
def model_pipeline():
    """Fixture to initialize the ModelPipeline."""
    windows = [3, 5]
    features_to_roll = ["value", "level"]
    diff_lags = [1]
    features_to_diff = ["value"]
    return ModelPipeline(
        windows=windows,
        features_to_roll=features_to_roll,
        diff_lags=diff_lags,
        features_to_diff=features_to_diff,
        groupby_col="provider",
    )


def test_cross_validation(mock_data, model_pipeline):
    """Test time-series cross-validation."""
    x, y = mock_data
    results = model_pipeline.cross_validate(x, y, n_splits=3)

    assert "test_f1" in results, "F1 score missing in cross-validation results."
    assert len(results["test_f1"]) == 3, "Cross-validation split count mismatch."


def test_model_serialization(mock_data, model_pipeline, tmp_path):
    """Test saving and loading the model pipeline."""
    x, y = mock_data
    save_name = "test_model"

    # Train and save the model
    model_pipeline.train(x, y)
    model_pipeline.save_model(save_name)

    # Mock the output path
    project_root = tmp_path / "output" / "models"
    project_root.mkdir(parents=True, exist_ok=True)

    # Load and verify predictions
    model_pipeline.load_model(save_name)
    predictions = model_pipeline.predict(x)
    assert len(predictions) == len(x), "Loaded model predictions length mismatch."


def test_invalid_inputs(model_pipeline):
    """Test invalid inputs for error handling."""
    with pytest.raises(ValueError, match="Input data must be a pandas DataFrame"):
        model_pipeline.predict(np.random.rand(10, 5))