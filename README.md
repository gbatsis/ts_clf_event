# 🚀 Machine Learning for Time Series Event Classification

<p align="center">
  <img src="/public/cover.png" alt="Project Cover Image" width="700" style="border-radius: 10px; box-shadow: 0 4px 8px rgba(0,0,0,0.1);"/>
</p>

> **Prompt:** A futuristic and abstract representation of machine learning for time series event classification. The image features a glowing neural network overlaying a dynamic time series graph with data points, highlighting patterns and trends. The background is dark with a blue and purple color scheme, giving it a high-tech and AI-driven feel. Subtle digital waves and matrix-like elements enhance the concept of data analysis and artificial intelligence.

<p align="center">
  <em>A robust machine learning pipeline for classifying events in time series data. From data preprocessing to a RESTful API.</em>
</p>

This project implements a comprehensive machine learning pipeline designed to classify events within time series datasets. It encompasses data exploration, feature engineering, model training with hyperparameter tuning, rigorous model evaluation, inference, and a RESTful API for seamless integration and prediction serving.

---

## 🛠️ Project Overview

This repository is structured to facilitate end-to-end machine learning for time-series event classification. Key components include:

-   **Data Exploration & Preprocessing:** Comprehensive analysis to understand data characteristics and prepare it for modeling.
-   **Feature Engineering:**  Extraction and creation of relevant features to enhance model performance.
-   **Model Training & Optimization:** Utilization of a Random Forest model with hyperparameter tuning using grid search.
-   **Model Evaluation:**  Detailed assessment of model performance using a variety of metrics and visualizations.
-   **Inference & API:**  Methods for running predictions and a RESTful API for real-time event classification.

---

## 📚 Methods and Results Summary

### 📊 1. Exploratory Data Analysis (EDA)

We adopted a rigorous approach to understanding the dataset, going beyond standard metrics:

-   **Initial Data Inspection:**  Examined sample rows, verified data types, and identified missing values to set the stage for analysis.
-   **Descriptive Statistics:** Computed mean, standard deviation, min/max, and quartiles to understand feature range and variability.
-   **Sampling Rate Analysis:** Inspected intervals between measurements to spot irregularities and inform windowing strategies.
-   **Visual Exploration:**  Plotted time series for features (temperature, level, etc.) segmented by provider and process status to capture trends.
-   **Distribution Checks:** Generated box plots to highlight nuances in feature distributions.
-   **Inter-feature Relationships:** Examined correlation matrices to identify linear dependencies between features.

> 🔗 **Dive Deeper:** [EDA Notebook](notebooks/01.%20EDA.ipynb)

---

### ⚙️ 2. Data Preprocessing and Feature Engineering

Key steps in preparing the data for modeling included:

-   **Chronological Data Split:** Partitioned the data into training and testing sets, holding out the last 2% for evaluation.
-   **Feature Engineering:**
    -   **Rolling Window Statistics:**  Computed moving averages, standard deviations, minima, and maxima using context-relevant window sizes.
    -   **Difference Calculations:** Derived lagged differences to capture dynamic changes.
    -   **Provider-Specific Adjustments:** Applied feature engineering steps individually to each provider for tailored processing.
-   **Scaling and Transformation:**  Normalized features using `MinMaxScaler` or `StandardScaler` to ensure no single feature dominates.

> 🔗 **Explore the Process:** [Feature Engineering Notebook](notebooks/02.%20Data%20processing.ipynb)

---

### 🧠 3. Model Development with Cross-Validation

We employed a Random Forest classifier for its robustness and ability to handle complex relationships.

-   **Hyperparameter Tuning:**  Used `GridSearchCV` to optimize hyperparameters, including `n_estimators`, `max_depth`, etc.
-   **Cross-Validation:** Performed time-series cross-validation (`TimeSeriesSplit`) to evaluate the model on temporally consistent data.
-   **Evaluation Metric:** Optimized for F1-score on the positive class (`f1_pos`) due to class imbalance.

> 🔗 **Detailed Model Training:** [Model Training Notebook](notebooks/03.%20Model_CV.ipynb)

**📊 Cross-Validation Results:** Details are available in `output/models/cv_results.pkl`.

---

### ✅ 4. Model Evaluation

The model was thoroughly evaluated on a held-out test set using various metrics:

-   **Metrics:** Precision, Recall, F1-score, Confusion Matrix, ROC Curve, AUC, MCC, Balanced Accuracy, Average Precision.

**📈 ROC-AUC and Precision-Recall Curves:**

<p align="center">
    <img src="/output/test_results/roc_curve.svg" alt="ROC-AUC Curve" width="450" style="border-radius: 5px; box-shadow: 0 2px 4px rgba(0,0,0,0.1);"/>
    <img src="/output/test_results/pr_curve.svg" alt="PR Curve" width="450" style="border-radius: 5px; box-shadow: 0 2px 4px rgba(0,0,0,0.1);"/>
</p>

---

## 🚀 Future Improvements for Production Deployment

Several enhancements could further optimize its performance and reliability in a production environment:

-   **Continuous Model Monitoring:** Implementing a system to monitor model performance in real-time, tracking metrics such as precision or recall. This would enable timely detection of degradation and trigger retraining or further analysis.
-   **Automated Retraining Pipelines:** Developing an automated retraining pipeline to periodically update the model with new data. This can be particularly important when the underlying time series patterns evolve or new event types emerge.
-   **A/B Testing:** Employing A/B testing for comparing different models and feature engineering strategies in production to ensure optimal performance on unseen data.
-   **Scalable Inference Service:** Optimizing the API service for scalability and high availability, ensuring low latency prediction serving even with high traffic loads.
-   **Advanced Model Selection:** Explore other models, such as Recurrent Neural Networks (RNNs) or Transformers, to potentially better capture complex time-dependencies of time series event data, and combine multiple model predictions with an ensembling approach.
-   **Data Quality Checks:** Implementing comprehensive data validation and quality checks to handle data inconsistencies, anomalies and missing values which can influence the model’s performance.
- **Error Analysis and Feedback Loop:** Establishing a feedback loop mechanism that incorporates identified misclassifications and their root cause analysis back into the model refinement process.

---

## 🚀 How to Run

### ⚙️ 1. Initial Setup

1.  **Clone the Repository:**

    ```bash
    git clone https://github.com/gbatsis/ts_clf_event.git
    cd ts_clf_event
    ```

2.  **Install Rye:**  Refer to the [Rye installation guide](https://rye.astral.sh/guide/installation/).

3.  **Data Preparation:**  Place data under `ts_clf_event/data/test_dataframe.csv`.

4.  **Dependency Installation:**

    ```bash
    rye sync
    ```

5.  **Activate the Environment:**

    ```bash
    source .venv/bin/activate
    ```

### 🚂 2. Training the Model

```bash
rye run manager train-hyper
```

*Note:*  Delete the `output` directory before executing to ensure grid search is run.

### 🧪 3. Evaluating the Model

```bash
rye run manager test
```

### 🔮 4. Running Inference

1. **Direct Module Invocation & Inference on the test data:**

   ```bash
    rye run manager inference mock
   ```

2.  **API Server:**

    ```bash
    uvicorn ts_clf_event.api.app:app --reload
    ```

    **Make a POST request:**

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

    **API Documentation:** [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)

3.  **Docker:**

    ```bash
    docker compose up
    ```

   **Inference on Test Data (Docker/API):**
   ```bash
   rye run manager inference mock-api
   ```

**We recommend to use API endpoints for inference in new data.**

---

## 📚 Documentation

**Run the Docs:**

```bash
mkdocs serve
```

- **Homepage:** The current `README.md`.
- **CLI Section:** Usage of commands via `rye.scripts` and `typer`.
- **Code Reference:** Explore project objects and functions.

---

**Note on Rye & CLI:** The project uses `rye.scripts` and `typer` to build the management CLI, explore the `src/ts_clf_event/cli` and the corresponsing section in the docs for more information.