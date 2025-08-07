import pandas as pd
from src.utils.evaluation import plot_metric_vs_rounds

# Load logs (assume CSVs saved by each experiment)
fedavg_log = pd.read_csv("results/fedavg_metrics.csv")
fedprox_log = pd.read_csv("results/fedprox_metrics.csv")

# Prepare metrics for plotting
metrics = {
    "FedAvg": fedavg_log["accuracy"].tolist(),
    "FedProx": fedprox_log["accuracy"].tolist(),
}

# Plot accuracy vs. communication rounds
plot_metric_vs_rounds(
    metrics,
    metric="accuracy",
    title="FedAvg vs FedProx: Accuracy on CIFAR-10 (Non-IID)",
    ylabel="Test Accuracy",
    save_path="results/accuracy_comparison.png"
)