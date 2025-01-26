# Data Pipeline

## **Time-based Data Split**
To simulate real-world scenarios, the dataset is split into training and test sets based on time.

```python
test_size_percent = 0.4
label_col = "process"

train_df, test_df = split_data_time_based(data_path, test_size_percent, label_col)

y_train = train_df[label_col]
y_test = test_df[label_col]

X_train = train_df.drop(label_col, axis=1)
X_test = test_df.drop(label_col, axis=1)
```

## **Pipeline Configuration**

The data pipeline involves:

1. Feature engineering:
   - Window-based rolling statistics (mean, std, min, max).
   - Difference features for specified lags.
2. Handling missing values using the training data statistics.
3. Scaling features to standardize the data.

Pipeline settings:
```python
windows = [30, 40, 60]
features_to_roll = ["value", "level", "frequency", "speed"]
diff_lags = [1, 2]
```

**Pipeline Initialization**: A custom data preprocessor is created to incorporate the above transformations.

```python
preprocessor = DataPreprocessor(
    features_to_roll=features_to_roll,
    features_to_diff=features_to_roll,
    windows=windows,
    diff_lags=diff_lags,
    groupby_col="provider",
)

pipeline = preprocessor.get_pipeline()
```

**Fit and Transform**: The pipeline is fitted on the training data and applied to both training and test datasets.

```python
pipeline.fit(X_train, y_train)

x_train_transformed = pipeline.transform(X_train)
x_test_transformed = pipeline.transform(X_test)
```

!!! info "More info in the corresponding [notebook](https://github.com/gbatsis/ts_clf_event/blob/feat-data_preprocessing/notebooks/02.%20Data%20processing.ipynb)."