import numpy as np
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset

def get_cifar10_transforms():
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

def load_cifar10_dataset(data_dir="data"):
    transform = get_cifar10_transforms()
    trainset = datasets.CIFAR10(root=data_dir, train=True, download=True, transform=transform)
    testset = datasets.CIFAR10(root=data_dir, train=False, download=True, transform=transform)
    return trainset, testset

def partition_data_non_iid(trainset, num_clients=10, classes_per_client=2, seed=42):
    np.random.seed(seed)
    labels = np.array(trainset.targets)
    class_indices = [np.where(labels == i)[0] for i in range(10)]
    client_indices = [[] for _ in range(num_clients)]
    all_classes = np.arange(10)
    for client in range(num_clients):
        chosen_classes = np.random.choice(all_classes, classes_per_client, replace=False)
        for cls in chosen_classes:
            idxs = np.random.choice(class_indices[cls], len(class_indices[cls]) // num_clients, replace=False)
            client_indices[client].extend(idxs)
            class_indices[cls] = np.setdiff1d(class_indices[cls], idxs)
    return client_indices

def partition_data_iid(trainset, num_clients=10, seed=42):
    np.random.seed(seed)
    indices = np.random.permutation(len(trainset))
    split = np.array_split(indices, num_clients)
    return [list(idx) for idx in split]

def get_client_loaders(trainset, testset, client_indices, batch_size=32):
    train_loaders = []
    for idxs in client_indices:
        loader = DataLoader(Subset(trainset, idxs), batch_size=batch_size, shuffle=True)
        train_loaders.append(loader)
    test_loader = DataLoader(testset, batch_size=batch_size, shuffle=False)
    return train_loaders, test_loader