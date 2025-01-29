## Inference Process Workflow Documentation

This document outlines the workflow for performing inference using the `Inference` class, which leverages a pre-trained time-series classification model to predict the probability of an event (positive class) based on new data points.

### 1. Overview

The inference process involves taking new data points, combining them with historical data for feature engineering, applying a pre-trained model pipeline to make predictions, and updating the historical data with the new data points. The process is designed to simulate a real-time scenario where new data arrives sequentially and predictions are made one at a time.

### 2. `Inference` Class

The `Inference` class handles the prediction process.

*   Initializes a `ModelPipeline` object with the specified feature engineering parameters.
*   Loads the pre-trained model from the specified file.

#### `predict` Method

1. **Convert Input to DataFrame:** Converts the input dictionary `data` to a pandas DataFrame.

2. **Handle Missing 'datetime' Column:** If the 'datetime' column is missing, it adds a 'datetime' column with the current timestamp. This is for time-based operations, although in the current implementation, the 'datetime' column might not be explicitly used in feature engineering.

3. **Combine with Historical Data:** Appends the new data point to the historical data (`self.history_df`). This is done to provide the necessary context for feature engineering, especially for rolling window and difference features.
    
4. **Predict Probability:** Calls the `predict_proba` method of the loaded model pipeline (`self.model`) to get the predicted probabilities for all data points in the combined DataFrame.
    
5. **Update Historical Data:** Appends the new data point to the historical data (`self.history_df`).
    
6. **Maintain History Window:** Keeps only the last `self.history_window` data points for each group (defined by `self.groupby_col`) in the historical data. This ensures that the historical data doesn't grow indefinitely and that it contains the most recent data relevant for feature engineering.
    
7. **Sort by Datetime:** Sorts the `self.history_df` by the 'datetime' column to maintain the correct time order.

8. **Return Predicted Probability:** Returns the predicted probability of the positive class for the last data point (which is the new data point) in the `y_pred` array.

### 3. `mock_inference` Function

Simulates an inference process using a pre-defined test dataset.

*   Splits the data into `history_df` (used as initial historical data) and `inference_df` (used as a stream of new data points) using `split_data_time_based`.

*   Initializes an `Inference` object with the historical data and feature engineering parameters.

*   Iterates through the `inference_df` DataFrame, simulating the arrival of new data points one at a time.

*   For each new data point:
    *   Calls the `predict` method to get the predicted probability.
    *   Stores the true label and predicted probability.
    *   Measures the prediction time.
    *   Prints the prediction and elapsed time for each row.

*   Uses the `Evaluator` class to generate a metrics report based on the collected true labels and predicted probabilities.

### 4. Logic Behind Using the Data

**Historical Data (`history_df`)**:

*   Used to provide context for feature engineering. Rolling window and difference features require a certain amount of historical data to be calculated.

*   Updated after each prediction to include the new data point, maintaining a rolling window of recent data.

*   The size of the historical window (`history_window`) should be chosen carefully based on the feature engineering parameters (maximum window size and lag) to ensure that sufficient data is available for calculations.

**Inference Data (`inference_df`)**:

*   Represents the stream of new data points for which predictions are to be made.

*   Processed one row at a time to simulate real-time inference.

*   The true labels (`process` column) are used to evaluate the performance of the model.

