# Federated Learning on CIFAR‑10: FedAvg vs. FedProx

## Overview

This repository contains a modular federated learning (FL) simulation built with Flower (flwr) and PyTorch on the CIFAR‑10 dataset. It focuses on training under heterogeneous, non‑IID client data to compare two FL algorithms:

- FedAvg (Federated Averaging)
- FedProx (Federated Proximal)

The codebase provides clean server/client apps, configurable experiments, and lightweight defaults for running on constrained machines.

## Highlights

- Flower Server/Client apps with a typed NumPyClient for CIFAR‑10.
- Non‑IID data partitioning and batched DataLoaders.
- Simple CNN baseline model for CIFAR‑10 (`SimpleCNN`).
- Config‑driven experiments for FedAvg and FedProx (YAML).
- Per‑round metrics aggregation and results logging to `results/history.{jsonl,csv}`.
- Memory‑friendly defaults and knobs (env vars) for local simulation.

## Tech stack

- Python (>=3.8), PyTorch, Torchvision
- Flower (flwr[simulation]) with Ray backend
- NumPy, pandas, matplotlib, seaborn, scikit‑learn

## Project structure

```
├── client_app.py               # Flower ClientApp (constructs clients per node)
├── server_app.py               # Flower ServerApp (strategy, rounds, logging)
├── run_simulation.py           # Simple Python API simulation runner (optional)
├── experiments/
│   ├── run_fedavg.py           # FedAvg experiment using config/fedavg_config.yaml
│   └── run_fedprox.py          # FedProx experiment using config/fedprox_config.yaml
├── config/
│   ├── fedavg_config.yaml
│   └── fedprox_config.yaml
├── src/
│   ├── models.py               # SimpleCNN model
│   └── utils/data_utils.py     # CIFAR‑10 loading, IID/non‑IID partitioning
├── results/                    # Auto‑created; stores history.jsonl and history.csv
├── requirements.txt            # Runtime dependencies (flwr[simulation], torch, ...)
├── pyproject.toml              # Package metadata + flwr app config
└── README.md
```

## Algorithms and components

- Model: `src/models.SimpleCNN` (two conv blocks + MLP head), suitable for CIFAR‑10.
- Data: `src/utils/data_utils.py`
	- `load_cifar10_dataset()` auto‑downloads to `data/`
	- `partition_data_non_iid(...)` for heterogeneous client splits
	- `get_client_loaders(...)` creates per‑client train loaders and a shared test loader
- Client: `client_app.CIFARClient` implements `get_parameters`, `fit`, `evaluate`.
- Server: `server_app.RecordingFedAvg` extends FedAvg to:
	- Aggregate metrics with weighted means.
	- Persist per‑round records to `results/history.jsonl` and `results/history.csv`.

## Setup

1) Create and activate a virtual environment, then install dependencies

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -U pip setuptools wheel
pip install -r requirements.txt
```

2) First run will auto‑download CIFAR‑10 into `data/`.

## Quickstart (Flower App)

Use the Flower App entrypoints defined in `pyproject.toml`:

```bash
source .venv/bin/activate
flwr run .
```

Memory‑friendly knobs (environment variables):

- `NUM_CLIENTS` (default 2)
- `BATCH_SIZE` (default 16)
- `NUM_ROUNDS` (default 5)
- `FRACTION_FIT`, `FRACTION_EVAL` (default 0.5)
- `MIN_FIT_CLIENTS`, `MIN_EVAL_CLIENTS`, `MIN_AVAILABLE_CLIENTS` (default 1)
- `RESULTS_DIR` (default `results`)

Example low‑resource run:

```bash
source .venv/bin/activate
NUM_CLIENTS=2 BATCH_SIZE=16 NUM_ROUNDS=2 \
FRACTION_FIT=0.5 FRACTION_EVAL=0.5 \
MIN_FIT_CLIENTS=1 MIN_EVAL_CLIENTS=1 MIN_AVAILABLE_CLIENTS=1 \
flwr run .
```

Note: `pyproject.toml` sets `options.num-supernodes = 2` to limit parallel Ray actors.

### Sample successful run output

When everything is set up correctly, a 5‑round low‑resource run typically looks like this:

```text
Loading project configuration... 
Success
INFO :      Starting Flower ServerApp, config: num_rounds=5, no round_timeout
INFO :      
INFO :      [INIT]
INFO :      Requesting initial parameters from one random client
INFO :      Received initial parameters from one random client
INFO :      Starting evaluation of initial global parameters
INFO :      Evaluation returned no results (`None`)
INFO :      
INFO :      [ROUND 1]
INFO :      configure_fit: strategy sampled 1 clients (out of 2)
INFO :      aggregate_fit: received 1 results and 0 failures
INFO :      configure_evaluate: strategy sampled 1 clients (out of 2)
INFO :      aggregate_evaluate: received 1 results and 0 failures
INFO :      
INFO :      [ROUND 2]
INFO :      configure_fit: strategy sampled 1 clients (out of 2)
INFO :      aggregate_fit: received 1 results and 0 failures
INFO :      configure_evaluate: strategy sampled 1 clients (out of 2)
INFO :      aggregate_evaluate: received 1 results and 0 failures
INFO :      
INFO :      [ROUND 3]
INFO :      configure_fit: strategy sampled 1 clients (out of 2)
INFO :      aggregate_fit: received 1 results and 0 failures
INFO :      configure_evaluate: strategy sampled 1 clients (out of 2)
INFO :      aggregate_evaluate: received 1 results and 0 failures
INFO :      
INFO :      [ROUND 4]
INFO :      configure_fit: strategy sampled 1 clients (out of 2)
INFO :      aggregate_fit: received 1 results and 0 failures
INFO :      configure_evaluate: strategy sampled 1 clients (out of 2)
INFO :      aggregate_evaluate: received 1 results and 0 failures
INFO :      
INFO :      [ROUND 5]
INFO :      configure_fit: strategy sampled 1 clients (out of 2)
INFO :      aggregate_fit: received 1 results and 0 failures
INFO :      configure_evaluate: strategy sampled 1 clients (out of 2)
INFO :      aggregate_evaluate: received 1 results and 0 failures
INFO :      
INFO :      [SUMMARY]
INFO :      Run finished 5 round(s)
INFO :          History (loss, distributed):
INFO :                  round 1: 9.971014263153076
INFO :                  round 2: 7.306167397689819
INFO :                  round 3: 10.769979529571533
INFO :                  round 4: 8.27235379562378
INFO :                  round 5: 11.324094764709473
INFO :          History (metrics, distributed, evaluate):
INFO :          {'accuracy': [(1, 0.1829), (2, 0.1698), (3, 0.1748), (4, 0.1856), (5, 0.1682)]}
```

## Running experiment scripts

FedAvg:

```bash
source .venv/bin/activate
python experiments/run_fedavg.py
```

FedProx:

```bash
source .venv/bin/activate
python experiments/run_fedprox.py
```

Edit `config/fedavg_config.yaml` / `config/fedprox_config.yaml` to control:

- `num_rounds`, `num_clients`, `clients_per_round`
- `local_epochs`, `batch_size`, `learning_rate`, `momentum`, `weight_decay`
- `mu` (FedProx proximal strength), `iid` flag, `seed`

## Alternative: simple simulation script

`run_simulation.py` shows a minimal Flower Python API simulation (10 clients, 5 rounds):

```bash
source .venv/bin/activate
python run_simulation.py
```

## Results and analysis

- Server writes per‑round aggregates to:
	- `results/history.jsonl` (one JSON record per round/phase)
	- `results/history.csv` (columns: round, phase, loss, metrics_json)
- Explore data/plots in the notebook: `notebooks/data_and_results_analysis.ipynb`.

## Tuning for limited memory

- Decrease `NUM_CLIENTS`, increase gradually.
- Lower `BATCH_SIZE` and number of rounds (`NUM_ROUNDS`).
- Keep `FRACTION_FIT/EVAL` small and `MIN_*_CLIENTS` minimal.
- `options.num-supernodes` in `pyproject.toml` limits Ray actors; set to a low value.

## Troubleshooting

- Ray memory/OOM: lower concurrency (env vars above) and supernodes; reduce batch size.
- PyTorch CUDA out‑of‑memory: run on CPU (unset CUDA) or reduce batch size.
- "No fit/evaluate metrics aggregation" warnings: resolved via server aggregation; ensure you run through `flwr run .` or use the updated `server_app.py`.

## License

This project is released under the MIT License. See `LICENSE` for details.

## Acknowledgements

- [Flower](https://flower.dev) for FL orchestration and simulation
- PyTorch/Torchvision for model and data tooling

#mm
