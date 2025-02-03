import os
import pandas as pd
import numpy as np
import plotly.graph_objects as go

from pathlib import Path
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    matthews_corrcoef,
    balanced_accuracy_score,
    average_precision_score,
    precision_recall_curve,
    roc_curve,
    confusion_matrix,
)

from ts_clf_event.utils.logging import setup_logger

logger = setup_logger()

class Evaluator:
    def __init__(self, output_dir=os.path.join(Path(__file__).parent.parent.parent.parent, "output", "test_results")):
        self.output_dir = output_dir
        if not os.path.isdir(self.output_dir):
            os.makedirs(self.output_dir)
        
    def report_metrics(
        self, true_labels: list, predicted_probs: list, threshold: float = 0.5) -> dict:
        """
        Generate and log a detailed metrics report, save results to files, and return metrics in a dictionary.

        Args:
            true_labels (list): Ground truth labels (0 or 1).
            predicted_probs (list): Model predictions (probabilities of the positive class).
            threshold (float, optional): Threshold for converting probabilities to binary
                                         predictions. Defaults to self.threshold.

        Returns:
            dict: A dictionary containing all calculated metrics.
        """

        # Convert probabilities to binary predictions
        predicted_labels = (np.array(predicted_probs) >= threshold).astype(int)

        # Calculate metrics
        metrics = {}
        metrics["precision_macro"] = precision_score(
            true_labels, predicted_labels, average="macro"
        )
        metrics["recall_macro"] = recall_score(
            true_labels, predicted_labels, average="macro"
        )
        metrics["f1_macro"] = f1_score(true_labels, predicted_labels, average="macro")
        metrics["precision_binary"] = precision_score(
            true_labels, predicted_labels, average="binary"
        )
        metrics["recall_binary"] = recall_score(
            true_labels, predicted_labels, average="binary"
        )
        metrics["f1_binary"] = f1_score(
            true_labels, predicted_labels, average="binary"
        )
        metrics["mcc"] = matthews_corrcoef(true_labels, predicted_labels)
        metrics["balanced_accuracy"] = balanced_accuracy_score(
            true_labels, predicted_labels
        )
        metrics["ap"] = average_precision_score(true_labels, predicted_probs)

        try:
            metrics["auroc"] = roc_auc_score(true_labels, predicted_probs)
        except ValueError:
            logger.error("Only one class present in y_true. ROC AUC score is not defined in that case.")
            
            metrics["auroc"] = np.nan

        precision, recall, pr_thresholds = precision_recall_curve(
            true_labels, predicted_probs
        )
        metrics["recall_at_precision_50"] = recall[np.argmin(np.abs(precision - 0.5))]

        fpr, tpr, roc_thresholds = roc_curve(true_labels, predicted_probs)

        # Confusion Matrix
        cm = confusion_matrix(true_labels, predicted_labels)
        cm_df = pd.DataFrame(
            cm,
            index=["Actual Negative", "Actual Positive"],
            columns=["Predicted Negative", "Predicted Positive"],
        )

        # Save confusion matrix to CSV
        cm_df.to_csv(os.path.join(self.output_dir, "confusion_matrix.csv"))

        # Plotting using Plotly
        self.plot_pr_curve(precision, recall, metrics["ap"])
        self.plot_roc_curve(fpr, tpr, metrics["auroc"])

        # Pretty - Print metrics with the logger
        logger.info("Confusion Matrix:\n%s", cm_df)
        logger.info("Macro-averaged Metrics:")
        logger.info("  Precision (Macro):  %s", metrics["precision_macro"])
        logger.info("  Recall (Macro):  %s", metrics["recall_macro"])
        logger.info("  F1-Score (Macro):  %s", metrics["f1_macro"])
        logger.info("Positive Class Metrics:")
        logger.info("  Precision ( Positive):  %s", metrics["precision_binary"])
        logger.info("  Recall ( Positive):  %s", metrics["recall_binary"])
        logger.info("  F1-Score ( Positive):  %s", metrics["f1_binary"])
        logger.info("Imbalance-Aware Metrics:")
        logger.info("  Matthews Correlation Coefficient (MCC):  %s", metrics["mcc"])
        logger.info("  Balanced Accuracy:  %s", metrics["balanced_accuracy"])
        logger.info("  Average Precision (AP):  %s", metrics["ap"])
        logger.info("  Area Under ROC Curve (AUROC):  %s", metrics["auroc"])
        logger.info("  Recall@Precision=0.5:  %s", metrics["recall_at_precision_50"])

        return metrics

    def plot_pr_curve(self, precision, recall, ap):
        """
        Plots the Precision-Recall curve using Plotly and saves it as an SVG file.

        Args:
            precision (np.ndarray): Array of precision values.
            recall (np.ndarray): Array of recall values.
            ap (float): Average precision score.
        """
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(x=recall, y=precision, mode="lines", name=f"AP = {ap:.2f}")
        )
        fig.update_layout(
            title="Precision-Recall Curve",
            xaxis_title="Recall",
            yaxis_title="Precision",
            template="plotly_white",
        )
        fig.write_image(os.path.join(self.output_dir, "pr_curve.svg"))

    def plot_roc_curve(self, fpr, tpr, auroc):
        """
        Plots the ROC curve using Plotly and saves it as an SVG file.

        Args:
            fpr (np.ndarray): Array of false positive rates.
            tpr (np.ndarray): Array of true positive rates.
            auroc (float): Area under the ROC curve.
        """
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines", name=f"AUROC = {auroc:.2f}"))
        fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", line=dict(dash="dash"), name="Chance"))
        fig.update_layout(
            title="Receiver Operating Characteristic (ROC) Curve",
            xaxis_title="False Positive Rate",
            yaxis_title="True Positive Rate",
            template="plotly_white",
        )
        fig.write_image(os.path.join(self.output_dir, "roc_curve.svg"))