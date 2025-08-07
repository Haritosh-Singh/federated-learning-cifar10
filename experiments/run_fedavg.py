
import yaml
import flwr as fl
import torch
import logging
from src.models import SimpleCNN
from src.utils.data_utils import load_cifar10_dataset, partition_data_non_iid, get_client_loaders

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Load config
try:
    with open("config/fedavg_config.yaml") as f:
        cfg = yaml.safe_load(f)
    logger.info("Loaded FedAvg config successfully.")
except Exception as e:
    logger.error(f"Failed to load config: {e}")
    raise


# Data partitioning
try:
    trainset, testset = load_cifar10_dataset()
    client_indices = partition_data_non_iid(trainset, num_clients=cfg["num_clients"])
    train_loaders, test_loader = get_client_loaders(trainset, testset, client_indices, batch_size=cfg["batch_size"])
    logger.info("Data partitioned and loaders created.")
except Exception as e:
    logger.error(f"Data loading/partitioning failed: {e}")
    raise


# Flower client function
def client_fn(cid):
    try:
        from src.client.client import CIFARClient
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = SimpleCNN().to(device)
        logger.info(f"Initialized client {cid}")
        return CIFARClient(model, train_loaders[int(cid)], test_loader, device)
    except Exception as e:
        logger.error(f"Client {cid} initialization failed: {e}")
        raise


# Start simulation
try:
    strategy = fl.server.strategy.FedAvg(
        fraction_fit=cfg["clients_per_round"]/cfg["num_clients"],
        min_fit_clients=cfg["clients_per_round"],
        min_available_clients=cfg["num_clients"],
        on_fit_config_fn=lambda rnd: {"local_epochs": cfg["local_epochs"], "learning_rate": cfg["learning_rate"]}
    )
    fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=cfg["num_clients"],
        config=fl.server.ServerConfig(num_rounds=cfg["num_rounds"]),
        strategy=strategy,
    )
    logger.info("FedAvg simulation completed successfully.")
except Exception as e:
    logger.error(f"FedAvg simulation failed: {e}")
    raise