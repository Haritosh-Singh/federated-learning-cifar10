import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def plot_metric_vs_rounds(metrics_dict, metric="accuracy", title="", ylabel="", save_path=None):
    """
    metrics_dict: dict, keys are algorithm names, values are lists of metric values per round
    """
    plt.figure(figsize=(8, 5))
    for algo, values in metrics_dict.items():
        plt.plot(values, label=algo)
    plt.title(title)
    plt.xlabel("Communication Round")
    plt.ylabel(ylabel or metric.capitalize())
    plt.legend()
    plt.grid(True)
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
    plt.show()

def aggregate_metrics(logs):
    """
    logs: list of dicts, each dict contains metrics for a round
    Returns: DataFrame with metrics per round
    """
    return pd.DataFrame(logs)

def save_metrics_csv(metrics_df, path):
    metrics_df.to_csv(path, index=False)