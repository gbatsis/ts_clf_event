# Machine Learning for event classification in time series

This project implements a machine learning pipeline for classifying events in time series data. It includes data preprocessing, feature engineering, model training with hyperparameter tuning, model evaluation, inference, and a RESTful API for serving predictions.

- [Machine Learning for event classification in time series](#machine-learning-for-event-classification-in-time-series)
  - [Methods and Results Summary](#methods-and-results-summary)
    - [1. Exploratory Data Analysis (EDA)](#1-exploratory-data-analysis-eda)
    - [2. Data Preprocessing and Feature Engineering](#2-data-preprocessing-and-feature-engineering)
    - [3. Model Development with Cross-Validation](#3-model-development-with-cross-validation)
    - [4. Model Evaluation](#4-model-evaluation)
  - [How to Run](#how-to-run)
    - [1. Initial Setup](#1-initial-setup)
    - [2. Training the Model](#2-training-the-model)
    - [3. Evaluating the Model](#3-evaluating-the-model)
    - [4. Running Inference](#4-running-inference)

**Examples & Tutorials:**  
> For a hands-on walkthrough, please refer to:

> - [EDA Notebook](notebooks/01.%20EDA.ipynb)

> - [Feature Engineering Notebook](notebooks/02.%20Data%20processing.ipynb)

> - [Model Training Notebook](notebooks/03.%20Model_CV.ipynb)

## Methods and Results Summary

### 1. Exploratory Data Analysis (EDA)

Before diving into model development, we spent time truly understanding our dataset. Rather than relying solely on standard metrics, we took a hands-on approach:

- **Initial Inspection:** We began by examining sample rows, verifying data types, and identifying any missing values. This helped set the stage for subsequent analysis.
- **Descriptive Overview:** Basic summary statistics (mean, standard deviation, min/max, quartiles) were computed to get a feel for the range and variability of each numerical feature.
- **Sampling Rate Analysis:** Given the time series nature of the data, we closely inspected the intervals between consecutive measurements. This was key in spotting irregular sampling rates and later informed our dynamic windowing strategy for different providers.
- **Visual Exploration:** Plotting the time series for features such as temperature, level, frequency, and speed allowed us to visually capture trends, cycles, and outliers. We segmented these plots by provider and process status (active/inactive) to better understand contextual differences.
- **Distribution Checks:** Box plots were generated for each feature (again, split by process status) to highlight distribution nuances.
- **Inter-feature Relationships:** Finally, we examined the correlation matrix to detect any linear dependencies among features, helping us decide on the best subsequent steps.

For an in-depth look at our exploratory process, please refer to the [EDA Notebook](notebooks/01.%20EDA.ipynb).

---

### 2. Data Preprocessing and Feature Engineering

- **Chronological Data Split:** To respect the temporal structure of the data, we partitioned it into training and testing sets based on time. The last 2% of records were held out to serve as a reliable gold standard.
- **Feature Engineering in Detail:**
  - **Rolling Window Statistics:** We computed moving averages, standard deviations, minima, and maxima for key features (`value`, `level`, `frequency`, `speed`). The window sizes were chosen based on the earlier sampling rate analysis, ensuring that the calculations were contextually relevant.
  - **Difference Calculations:** To capture the dynamics of change over time, we derived lagged differences for the selected features.
  - **Provider-Specific Adjustments:** Recognizing that each provider’s data could behave differently, we applied these feature engineering steps on a per-provider basis. This helped tailor the feature space to account for varied sampling rates and data characteristics.
- **Scaling and Transformation:** Finally, to mitigate the risk of any single feature dominating the model due to scale differences, we applied normalization techniques (using either `MinMaxScaler` or `StandardScaler` as appropriate).

For the complete set of preprocessing steps and feature engineering details, the [Feature Engineering Notebook](notebooks/02.%20Data%20processing.ipynb) offers a comprehensive walkthrough.

### 3. Model Development with Cross-Validation

A Random Forest classifier was chosen for this project due to its ability to handle complex relationships, its robustness to overfitting, and its built-in feature importance estimation.

*   **Hyperparameter Tuning:** `GridSearchCV` was used to find the optimal hyperparameters for the Random Forest model. The following hyperparameters were tuned:
    *   `n_estimators`: Number of trees in the forest.
    *   `max_depth`: Maximum depth of each tree.
    *   `min_samples_split`: Minimum number of samples required to split an internal node.
    *   `min_samples_leaf`: Minimum number of samples required to be at a leaf node.
    *   `class_weight`: Weights associated with each class (to address class imbalance).
*   **Cross-Validation:** `TimeSeriesSplit` was used to perform time-series cross-validation, ensuring that the model was evaluated on data that comes after the training data in each fold. This approach respects the temporal nature of the data and provides a more realistic estimate of the model's performance on unseen data.
*   **Evaluation Metric:** The F1-score for the positive class (`f1_pos`) was used as the primary metric to optimize during hyperparameter tuning, considering the importance of correctly identifying positive events and the presence of class imbalance.

**Notebook:** The model development and training process are documented in the [Model Training Notebook](notebooks/03.%20Model_CV.ipynb).

**Cross-Validation Results:** The detailed cross-validation results, including the performance of each hyperparameter combination, are saved in the `output/models/cv_results.pkl` file.

### 4. Model Evaluation

The trained model was evaluated on a held-out test set to assess its performance on unseen data. The following metrics were used:

*   **Precision:** The proportion of true positive predictions among all positive predictions.
*   **Recall:** The proportion of true positive predictions among all actual positive instances.
*   **F1-score:** The harmonic mean of precision and recall.
*   **Confusion Matrix:** A table showing the counts of true positives, true negatives, false positives, and false negatives.
*   **ROC Curve:** A plot of the true positive rate against the false positive rate at various thresholds.
*   **AUC (Area Under the ROC Curve):** A measure of the model's ability to distinguish between the two classes.
*   **Matthews Correlation Coefficient (MCC):** A measure that takes into account all four values in the confusion matrix, making it suitable for imbalanced datasets.
*   **Balanced Accuracy:** The average of recall obtained on each class.
*   **Average Precision (AP):** The average precision across all thresholds.

**ROC-AUC and Precision-Recall Curves:**

![ROC-AUC Curve](/output/test_results/roc_curve.svg)

![PR Curve](/output/test_results/pr_curve.svg)

## How to Run

### 1. Initial Setup

1. **Clone the Repository:**  
   Start by cloning the repository to your local machine.

2. **Install Rye:**  
   This project leverages [Rye](https://rye.astral.sh/guide/installation/) for comprehensive project and package management. Please refer to the installation guide if you haven’t installed it already.

3. **Data Preparation:**  
   Place the required data in the designated location as specified by the project configuration. Under the `ts_clf_event/data/test_dataframe.csv`.

4. **Dependency Installation:**  
   Run the following command to sync the environment, create the `.env` file, and install all necessary dependencies:
   ```bash
   rye sync
   ```

5. **Activate the Environment:**  
   Once the dependencies are installed, activate your virtual environment:
   ```bash
   source .venv/bin/activate
   ```

   *Note:* We utilize `rye.scripts` along with `typer` to build a management CLI for this project. You can explore these tools under the `src/ts_clf_event/cli` module.

### 2. Training the Model
To identify the optimal hyperparameters via grid-search CV and subsequently train the model on the entire training set (with the trained model saved in the `output/models` directory), execute:

```bash
rye run manager train-hyper
```

*Note:* If the `output` directory already exists, make sure to delete it before running this command to ensure the grid-search CV step is executed.

### 3. Evaluating the Model

After training, assess the model’s performance on the held-out test set using:

```bash
rye run manager test
```

### 4. Running Inference

You have a couple of options to perform inference:

1. **Direct Module Invocation:**  
   Use the `inference` module directly or refer to the `mock` function in `src/ts_clf_event/cli/commands/inference.py` for an example using the test subset as mock data.

2. **Using the API:**  
   Start the API server with:
   ```bash
   uvicorn ts_clf_event.api.app:app --reload
   ```
   Then, make an API call using `curl`:
   ```bash
   curl -X 'POST' \
     'http://127.0.0.1:8000/predict/' \
     -H 'accept: application/json' \
     -H 'Content-Type: application/json' \
     -d '{
       "start_value": 0,
       "value": 0,
       "speed": 0,
       "level": 0,
       "frequency": 0,
       "status": 0,
       "datetime": "string",
       "provider": "string"
     }'
   ```
   Alternatively, you can interact with the API using Swagger UI at:  
   [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)

3. **Using Docker:**  
   If you prefer containerization, simply run:
   ```bash
   docker compose up
   ```

In both scenarios, you can also perform inference on the test data by executing:
```bash
rye run mock-api
```