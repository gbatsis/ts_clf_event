import numpy as np

def display_cv_results(cv_results: dict) -> None:
    """
    Pretty-print the cross-validation results.

    Args:
        cv_results (dict): A dictionary containing cross-validation results.
    """
    # Extract metrics and compute mean and standard deviation
    metrics = {
        "precision": {
            "mean": np.mean(cv_results["test_precision"]),
            "std": np.std(cv_results["test_precision"]),
        },
        "recall": {
            "mean": np.mean(cv_results["test_recall"]),
            "std": np.std(cv_results["test_recall"]),
        },
        "f1": {
            "mean": np.mean(cv_results["test_f1"]),
            "std": np.std(cv_results["test_f1"]),
        },
        "precision_pos": {
            "mean": np.mean(cv_results["test_precision_pos"]),
            "std": np.std(cv_results["test_precision_pos"]),
        },
        "recall_pos": {
            "mean": np.mean(cv_results["test_recall_pos"]),
            "std": np.std(cv_results["test_recall_pos"]),
        },
        "f1_pos": {
            "mean": np.mean(cv_results["test_f1_pos"]),
            "std": np.std(cv_results["test_f1_pos"]),
        },
    }

    # Print the results in a tabular format
    print("{:<15} {:<10} {:<10}".format("Metric", "Mean", "Std"))
    print("-" * 35)
    for metric, values in metrics.items():
        print("{:<15} {:<10.4f} {:<10.4f}".format(metric, values["mean"], values["std"]))