# CLI Documentation: Step-by-Step Guide

This guide walks you through how the Command Line Interface (CLI) is implemented in the project. It covers the process of initializing the CLI with Typer, and details each command group for training, evaluating, and performing inference. Each section includes code snippets to highlight important parts of the implementation.

---

## 1. Initializing the CLI

The CLI is built using [Typer](https://typer.tiangolo.com/), which simplifies the creation of command-line applications by automatically generating help messages and validating inputs.

**Key Steps:**

- **Import Typer and Command Groups:**  
  The main script imports the command groups for training, evaluation, and inference.

- **Register Command Groups:**  
  The command groups are added to the main Typer app. Trainer and evaluator commands are registered at the root, while inference commands are nested under the `inference` subcommand.

**Main Entry Point Code:**

```python
import typer

from ts_clf_event.cli.commands import trainer_cli, evaluator_cli, inference_cli

app = typer.Typer()
app.add_typer(trainer_cli, name="")           # Trainer commands at the root
app.add_typer(evaluator_cli, name="")          # Evaluator commands at the root
app.add_typer(inference_cli, name="inference")   # Inference commands nested under 'inference'

if __name__ == "__main__":
    app()
```

---

## 2. Trainer CLI Command

The **Trainer CLI** is responsible for model training, including hyperparameter tuning. The `train_hyper` command is designed to split the dataset, tune hyperparameters with grid-search cross-validation, re-train the model using the best parameters, and then save the trained model.

**Step-by-Step Process:**

1. **Data Preparation:**  
   Load and split the data in a time-based manner to maintain the temporal order.  
2. **Feature Engineering:**  
   Set up rolling window features and lag differences.  
3. **Hyperparameter Tuning:**  
   Run grid-search CV to identify the best parameters.
4. **Re-Training & Saving the Model:**  
   Train the model using the entire training set and save it.

**Code Snippet for `train_hyper`:**

```python
import os
import typer
from pathlib import Path

from ts_clf_event.data_handler.utils import split_data_time_based
from ts_clf_event.model.model import ModelPipeline
from ts_clf_event.utils.logging import setup_logger

logger = setup_logger()
DATA_PATH = os.path.join(Path(__file__).parent.parent.parent.parent.parent, "data", "test_dataframe.csv")

trainer_cli: typer.Typer = typer.Typer()

@trainer_cli.command()
def train_hyper():
    """
    Trains a model after hyperparameter tuning.
    """
    test_size_percent = 0.2
    label_col = "process"

    # Step 1: Split the data while preserving temporal order
    train_df, _ = split_data_time_based(DATA_PATH, test_size_percent, label_col)
    
    # Feature Engineering Settings
    windows = "auto" 
    features_to_roll = ["value", "level", "frequency", "speed"]
    diff_lags = [1, 2]
    
    # Prepare training data
    x_train = train_df.drop("process", axis=1)
    y_train = train_df["process"]

    # Step 2: Initialize the model pipeline with feature settings
    model_pipeline = ModelPipeline(
        windows=windows,
        features_to_roll=features_to_roll,
        diff_lags=diff_lags,
        features_to_diff=features_to_roll,
        groupby_col="provider",
        params=None
    )

    # Step 3: Perform hyperparameter tuning via grid-search CV
    model_pipeline.train(
        x_train,
        y_train,
        tune_hyperparameters=True,
        n_splits=5,
    )
    logger.info(model_pipeline.process_cv_results(model_pipeline.cv_results))

    # Step 4: Re-train with the best hyperparameters and save the model
    model_pipeline.train(
        x_train,
        y_train,
        tune_hyperparameters=False,
    )
    model_pipeline.save_model("RF_model")
```

---

## 3. Evaluator CLI Command

The **Evaluator CLI** focuses on testing a pre-trained model. The `test` command loads the saved model, applies it to the test dataset, and reports performance metrics.

**Step-by-Step Process:**

1. **Data Splitting:**  
   Separate the test set from the dataset.
2. **Model Loading:**  
   Load the previously saved model.
3. **Model Prediction:**  
   Generate prediction probabilities for the test set.
4. **Evaluation:**  
   Use an evaluator to report metrics such as accuracy, precision, recall, etc.

**Code Snippet for `test`:**

```python
import os
import typer
from pathlib import Path

from ts_clf_event.data_handler.utils import split_data_time_based
from ts_clf_event.model.model import ModelPipeline
from ts_clf_event.model.evaluator import Evaluator

DATA_PATH = os.path.join(Path(__file__).parent.parent.parent.parent.parent, "data", "test_dataframe.csv")

evaluator_cli: typer.Typer = typer.Typer()

@evaluator_cli.command()
def test():
    """
    Evaluates an existing model.
    """
    test_size_percent = 0.2
    label_col = "process"

    # Step 1: Load the test dataset
    _, test_df = split_data_time_based(DATA_PATH, test_size_percent, label_col)
    
    # Feature Engineering Settings (consistent with training)
    windows = "auto" 
    features_to_roll = ["value", "level", "frequency", "speed"]
    diff_lags = [1, 2]

    # Step 2: Load the model pipeline
    model_pipeline = ModelPipeline(
        windows=windows,
        features_to_roll=features_to_roll,
        diff_lags=diff_lags,
        features_to_diff=features_to_roll,
        groupby_col="provider",
        params=None
    )
    model_pipeline.load_model("RF_model")
    
    # Prepare test data and generate predictions
    x_test = test_df.drop("process", axis=1)
    y_test = test_df["process"]
    y_pred_prob = model_pipeline.predict_proba(x_test)

    # Step 3: Evaluate the predictions
    Evaluator().report_metrics(y_test, y_pred_prob[:, 1], threshold=0.5)
```

---

## 4. Inference CLI Commands

The inference commands enable real-time and API-based inference using the test subset (1 data point per time). Two main commands are provided: `mock` and `mock_api`.

### 4.1 Mock Inference

The `mock` command simulates real-time inference using a test subset of the data. It uses a historical window for context, iterates over test instances, and logs the prediction time for each.

**Step-by-Step Process:**

1. **Data Preparation:**  
   Split the dataset and select a history window.
2. **Initialize Inference Object:**  
   Configure inference settings using the same feature engineering as during training.
3. **Iterate and Predict:**  
   For each data point, generate predictions and log performance.
4. **Report Evaluation Metrics:**  
   Evaluate aggregated predictions.

**Code Snippet for `mock`:**

```python
import os
import time
import typer
from pathlib import Path
from ts_clf_event.data_handler.utils import split_data_time_based
from ts_clf_event.inference.inference import Inference
from ts_clf_event.model.evaluator import Evaluator
from ts_clf_event.utils.logging import setup_logger

logger = setup_logger()
DATA_PATH = os.path.join(Path(__file__).parent.parent.parent.parent.parent, "data", "test_dataframe.csv")
HISTORY_WINDOW = 1000

inference_cli: typer.Typer = typer.Typer()

@inference_cli.command()
def mock():
    """
    Mock inference for testing purposes.

    This command splits the dataset, uses the training portion as a history window,
    and performs inference on the test subset.
    """
    test_size_percent = 0.2
    label_col = "process"

    # Step 1: Split data into history and inference portions
    history_df, inference_df = split_data_time_based(DATA_PATH, test_size_percent, label_col)
    windows = "auto"
    features_to_roll = ["value", "level", "frequency", "speed"]
    diff_lags = [1, 2]
    group_by_col = "provider"

    # Keep only the last HISTORY_WINDOW rows per group and sort
    history_df = history_df.groupby(group_by_col).tail(HISTORY_WINDOW)
    history_df = history_df.reset_index(drop=True).sort_values("datetime")
    history_df = history_df.drop("process", axis=1)

    # Step 2: Initialize the inference object
    inference = Inference(history_df, windows, features_to_roll, diff_lags, features_to_roll, group_by_col, HISTORY_WINDOW)
    
    y_true_list = []
    y_pred_list = []

    # Step 3: Process each data point for inference
    for index, row in inference_df.iterrows():
        start_time = time.time()
        data = row.drop("process").to_dict()
        y_pred_prob = inference.predict(data)
        y_true_list.append(row["process"])
        y_pred_list.append(y_pred_prob)
        elapsed_time = time.time() - start_time
        logger.info(f"Predicted probability: {y_pred_prob} in {elapsed_time} seconds")

    # Step 4: Report overall evaluation metrics
    evaluator = Evaluator("./output/mock_inf_results")
    evaluator.report_metrics(y_true_list, y_pred_list)
```

### 4.2 API-Based Inference

The `mock_api` command performs inference by sending HTTP POST requests to a running API server. This command demonstrates how the inference can be integrated into a production-like environment.

**Step-by-Step Process:**

1. **Prepare Data:**  
   Use the test subset for inference.
2. **Send HTTP Request:**  
   For each instance, send a POST request with the data to the API endpoint.
3. **Collect and Log Predictions:**  
   Record the predicted probability and the time taken for each request.
4. **Evaluate Performance:**  
   Aggregate and evaluate the predictions.

**Code Snippet for `mock_api`:**

```python
import os
import time
import requests
import typer
from pathlib import Path
from ts_clf_event.data_handler.utils import split_data_time_based
from ts_clf_event.model.evaluator import Evaluator
from ts_clf_event.utils.logging import setup_logger

logger = setup_logger()
DATA_PATH = os.path.join(Path(__file__).parent.parent.parent.parent.parent, "data", "test_dataframe.csv")

inference_cli: typer.Typer = typer.Typer()

@inference_cli.command()
def mock_api():
    """
    Mock inference for testing purposes using the API.

    This command sends POST requests to the API endpoint for each data point in the test subset.
    """
    test_size_percent = 0.2
    label_col = "process"

    # Step 1: Split data; only use the test subset for API inference
    _, inference_df = split_data_time_based(DATA_PATH, test_size_percent, label_col)
    
    y_true_list = []
    y_pred_list = []

    # Step 2: Loop over each test data point and make an API request
    for index, row in inference_df.iterrows():
        start_time = time.time()
        data = row.drop("process").to_dict()
        url = "http://127.0.0.1:8000/predict/"
        response = requests.post(url, json=data)
        y_pred_prob = response.json()['probability']
        y_true_list.append(row["process"])
        y_pred_list.append(y_pred_prob)
        elapsed_time = time.time() - start_time
        logger.info(f"Predicted probability: {y_pred_prob} in {elapsed_time} seconds")
        
    # Step 3: Evaluate the API-based predictions
    evaluator = Evaluator("./output/mock_inf_results")
    evaluator.report_metrics(y_true_list, y_pred_list)
```

---

This CLI implementation provides a modular and extensible way to manage your machine learning workflows. The clear separation between training, evaluation, and inference allows you to easily update or extend functionality as needed. By following this step-by-step guide and referring to the code snippets provided, you can gain a deeper understanding of how the CLI is structured and how each command contributes to the overall process. For further details or to modify the CLI behavior, please refer to the source code in the `src/ts_clf_event/cli/commands` directory.

