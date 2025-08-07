
import flwr as fl
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
import numpy as np
import sys
import os

# Import the model
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from models import SimpleCNN

# Data loading and partitioning utilities
def load_partition(partition, batch_size=32):
	"""
	Loads a partition of CIFAR-10 for a given client index.
	"""
	transform = transforms.Compose([
		transforms.ToTensor(),
		transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
	])
	trainset = datasets.CIFAR10(root="data", train=True, download=True, transform=transform)
	testset = datasets.CIFAR10(root="data", train=False, download=True, transform=transform)

	# For demonstration, use a simple partitioning: each client gets every Nth sample
	num_clients = 10
	idxs = np.arange(len(trainset))
	client_idxs = idxs[partition::num_clients]
	train_subset = Subset(trainset, client_idxs)
	train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
	test_loader = DataLoader(testset, batch_size=batch_size, shuffle=False)
	return train_loader, test_loader

# Flower client implementation
class CIFARClient(fl.client.NumPyClient):
	def __init__(self, model, train_loader, test_loader, device):
		self.model = model
		self.train_loader = train_loader
		self.test_loader = test_loader
		self.device = device

	def get_parameters(self, config=None):
		return [val.cpu().numpy() for val in self.model.state_dict().values()]

	def set_parameters(self, parameters):
		params_dict = zip(self.model.state_dict().keys(), parameters)
		state_dict = {k: torch.tensor(v) for k, v in params_dict}
		self.model.load_state_dict(state_dict, strict=True)

	def fit(self, parameters, config):
		self.set_parameters(parameters)
		self.model.train()
		optimizer = optim.SGD(self.model.parameters(), lr=0.01, momentum=0.9)
		criterion = nn.CrossEntropyLoss()
		epochs = 1  # For simulation, keep it small
		for _ in range(epochs):
			for data, target in self.train_loader:
				data, target = data.to(self.device), target.to(self.device)
				optimizer.zero_grad()
				output = self.model(data)
				loss = criterion(output, target)
				loss.backward()
				optimizer.step()
		return self.get_parameters(), len(self.train_loader.dataset), {}

	def evaluate(self, parameters, config):
		self.set_parameters(parameters)
		self.model.eval()
		criterion = nn.CrossEntropyLoss()
		loss = 0
		correct = 0
		with torch.no_grad():
			for data, target in self.test_loader:
				data, target = data.to(self.device), target.to(self.device)
				output = self.model(data)
				loss += criterion(output, target).item() * data.size(0)
				pred = output.argmax(dim=1, keepdim=True)
				correct += pred.eq(target.view_as(pred)).sum().item()
		loss /= len(self.test_loader.dataset)
		accuracy = correct / len(self.test_loader.dataset)
		return float(loss), len(self.test_loader.dataset), {"accuracy": float(accuracy)}


def main():
	partition = int(os.environ.get("CLIENT_ID", 0))  # Set by Flower simulation
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	model = SimpleCNN().to(device)
	train_loader, test_loader = load_partition(partition)
	client = CIFARClient(model, train_loader, test_loader, device)
	fl.client.start_numpy_client(server_address="localhost:8080", client=client)

if __name__ == "__main__":
	main()
