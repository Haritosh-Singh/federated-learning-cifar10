import yaml
import flwr as fl
import torch
from src.models import SimpleCNN
from src.utils.data_utils import load_cifar10_dataset, partition_data_non_iid, get_client_loaders

# Load config
with open("config/fedprox_config.yaml") as f:
    cfg = yaml.safe_load(f)

# Data partitioning
trainset, testset = load_cifar10_dataset()
client_indices = partition_data_non_iid(trainset, num_clients=cfg["num_clients"])
train_loaders, test_loader = get_client_loaders(trainset, testset, client_indices, batch_size=cfg["batch_size"])

# Flower client function
def client_fn(cid):
    from src.client.fedprox_client import FedProxClient
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SimpleCNN().to(device)
    return FedProxClient(model, train_loaders[int(cid)], test_loader, device, mu=cfg["mu"])

# Start simulation
strategy = fl.server.strategy.FedAvg(
    fraction_fit=cfg["clients_per_round"]/cfg["num_clients"],
    min_fit_clients=cfg["clients_per_round"],
    min_available_clients=cfg["num_clients"],
    on_fit_config_fn=lambda rnd: {"local_epochs": cfg["local_epochs"], "learning_rate": cfg["learning_rate"], "mu": cfg["mu"]}
)
fl.simulation.start_simulation(
    client_fn=client_fn,
    num_clients=cfg["num_clients"],
    config=fl.server.ServerConfig(num_rounds=cfg["num_rounds"]),
    strategy=strategy,
)