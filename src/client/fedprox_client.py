
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

# Data loading and partitioning utilities (same as FedAvg)
def load_partition(partition, batch_size=32):
	transform = transforms.Compose([
		transforms.ToTensor(),
		transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
	])
	trainset = datasets.CIFAR10(root="data", train=True, download=True, transform=transform)
	testset = datasets.CIFAR10(root="data", train=False, download=True, transform=transform)
	num_clients = 10
	idxs = np.arange(len(trainset))
	client_idxs = idxs[partition::num_clients]
	train_subset = Subset(trainset, client_idxs)
	train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
	test_loader = DataLoader(testset, batch_size=batch_size, shuffle=False)
	return train_loader, test_loader

# FedProx Flower client
class FedProxClient(fl.client.NumPyClient):
	def __init__(self, model, train_loader, test_loader, device, mu=0.01):
		self.model = model
		self.train_loader = train_loader
		self.test_loader = test_loader
		self.device = device
		self.mu = mu
		self.global_params = None

	def get_parameters(self, config=None):
		return [val.cpu().numpy() for val in self.model.state_dict().values()]

	def set_parameters(self, parameters):
		params_dict = zip(self.model.state_dict().keys(), parameters)
		state_dict = {k: torch.tensor(v) for k, v in params_dict}
		self.model.load_state_dict(state_dict, strict=True)

	def fit(self, parameters, config):
		self.set_parameters(parameters)
		self.global_params = [p.clone().detach() for p in self.model.parameters()]
		self.model.train()
		optimizer = optim.SGD(self.model.parameters(), lr=0.01, momentum=0.9)
		criterion = nn.CrossEntropyLoss()
		epochs = 1
		mu = self.mu
		for _ in range(epochs):
			for data, target in self.train_loader:
				data, target = data.to(self.device), target.to(self.device)
				optimizer.zero_grad()
				output = self.model(data)
				loss = criterion(output, target)
				# FedProx proximal term
				prox_reg = 0.0
				for w, w0 in zip(self.model.parameters(), self.global_params):
					prox_reg += ((w - w0).norm(2)) ** 2
				loss += (mu / 2) * prox_reg
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
	partition = int(os.environ.get("CLIENT_ID", 0))
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	model = SimpleCNN().to(device)
	train_loader, test_loader = load_partition(partition)
	mu = float(os.environ.get("FEDPROX_MU", 0.01))
	client = FedProxClient(model, train_loader, test_loader, device, mu=mu)
	fl.client.start_numpy_client(server_address="localhost:8080", client=client)

if __name__ == "__main__":
	main()
