import os
import requests
import configparser
from typing import List, Tuple
import pickle
import fnmatch
import re

from flwr.common import NDArrays, Scalar, ndarrays_to_parameters, Parameters, FitRes
from flwr.server.client_proxy import ClientProxy
import numpy as np
from typing import Dict, Optional, Tuple, Union

import flwr as fl
import tensorflow as tf

from flwr.server.client_manager import SimpleClientManager
from FedAvg import FedAvgCustom
from FedProx import FedProxCustom
from QFedAvg import QFedAvgCustom
from FedOpt import FedOptCustom

# Configuration file path
config_path = os.environ.get('CONFIG_PATH')
config = configparser.ConfigParser()
config.read(config_path)

# Variables from the configuration file
num_exec = config.get('configVariable', 'num_exec')
strategy_name = config.get('configVariable', 'strategy')
rounds = config.getint('configVariable', 'step_rounds')

checkpoint_path = os.path.expanduser(config.get('configPaths', 'checkpoint').format(strategy=strategy_name, num_exec=config.getint('configVariable', 'num_exec')))

model = tf.keras.applications.MobileNet((32, 32, 1), classes=10, weights=None)
model.compile("adam", "sparse_categorical_crossentropy", metrics=["accuracy"])

# Function for calculating the weighted average metric
def fit_weighted_average(
        metrics: List[Tuple[int, fl.common.Metrics]]) -> fl.common.Metrics:

    accuracies = [num_examples * m["accuracy"] for num_examples, m in metrics]
    losses = [num_examples * m["loss"] for num_examples, m in metrics]
    losses_distributed = [num_examples * m["loss_distributed"] for num_examples, m in metrics]
    recalls = [num_examples * m["recall"] for num_examples, m in metrics]
    precisions = [num_examples * m["precision"] for num_examples, m in metrics]
    f1_scores = [num_examples * m["f1_score"] for num_examples, m in metrics]

    examples = [num_examples for num_examples, _ in metrics]

    return {"accuracy": sum(accuracies) / sum(examples), "loss": sum(losses) / sum(examples), "loss_distributed": sum(losses_distributed) / sum(examples), "recall": sum(recalls) / sum(examples), "precision": sum(precisions) / sum(examples), "f1_score": sum(f1_scores) / sum(examples)}

# Function for calculating the weighted average metric
def evaluate_weighted_average(
        metrics: List[Tuple[int, fl.common.Metrics]]) -> fl.common.Metrics:

    accuracies = [num_examples * m["accuracy"] for num_examples, m in metrics]
    recalls = [num_examples * m["recall"] for num_examples, m in metrics]
    precisions = [num_examples * m["precision"] for num_examples, m in metrics]
    f1_scores = [num_examples * m["f1_score"] for num_examples, m in metrics]

    examples = [num_examples for num_examples, _ in metrics]

    return {"accuracy": sum(accuracies) / sum(examples), "recall": sum(recalls) / sum(examples), "precision": sum(precisions) / sum(examples), "f1_score": sum(f1_scores) / sum(examples)}

# Dictionary mapping names to functions
functions_dict = {
    'server_side': ndarrays_to_parameters(model.get_weights()),
    'fit_weighted_average': fit_weighted_average,
    'evaluate_weighted_average': evaluate_weighted_average,
    'None': None,
}

# Configuration extracted from the configuration file
configurations = {
    'fraction_fit': config.getfloat('configFixed', 'fraction_fit'),
    'fraction_evaluate': config.getfloat('configFixed', 'fraction_evaluate'),
    'min_fit_clients': config.getint('configFixed', 'min_fit_clients'),
    'min_evaluate_clients': config.getint('configFixed', 'min_evaluate_clients'),
    'min_available_clients': config.getint('configFixed', 'min_available_clients'),
    'evaluate_fn': config.get('configFixed', 'evaluate_fn'),
    'on_fit_config_fn': config.get('configFixed', 'on_fit_config_fn'),
    'on_evaluate_config_fn': config.get('configFixed', 'on_evaluate_config_fn'),
    'accept_failures': config.getboolean('configFixed', 'accept_failures'),
    'initial_parameters': config.get('configFixed', 'initial_parameters'),
    'fit_metrics_aggregation_fn': config.get('configFixed', 'fit_metrics_aggregation_fn'),
    'evaluate_metrics_aggregation_fn': config.get('configFixed', 'evaluate_metrics_aggregation_fn'),
    'num_exec': num_exec,
    'strategy_name': strategy_name,
}

# Map values from configuration to functions based on the dictionary
for key, value in configurations.items():
    if value in functions_dict:
        configurations[key] = functions_dict[value]

# Create strategy based on the strategy selected in the configuration file
def fedAvg():
    strategy = FedAvgCustom(**configurations)
    return strategy


def fedProx():
    strategy = FedProxCustom(**configurations,
                            proximal_mu=config.getfloat('configFedProx', 'proximal_mu'),
                            )
    return strategy


def qFedAvg():
    strategy = QFedAvgCustom(**configurations,
                            q_param=config.getfloat('configQFedAvg', 'q_param'),
                            qffl_learning_rate=config.getfloat('configQFedAvg', 'qffl_learning_rate'),
                            )
    return strategy

def fedOpt():
    strategy = FedOptCustom(**configurations,
                            eta=config.getfloat('configFedYogi', 'eta'),
                            eta_l=config.getfloat('configFedYogi', 'eta_l'),
                            beta_1=config.getfloat('configFedYogi', 'beta_1'),
                            beta_2=config.getfloat('configFedYogi', 'beta_2'),
                            tau=config.getfloat('configFedYogi', 'tau'),
                            )
    return strategy


# Default function if the strategy is not selected correctly
def default():
    return "Nothing selected"


# Map strategy names to functions
switch = {
    "FedAvg": fedAvg,
    "FedProx": fedProx,
    "QFedAvg": qFedAvg,
    "FedOpt": fedOpt,
}

# Select strategy based on the name provided in the configuration file
selected_strategy = switch.get(strategy_name, default)
strategy = selected_strategy()

# Configuration to delete data series on the server
url = 'http://localhost:9090/api/v1/admin/tsdb/delete_series'
params = {'match[]': '{job="cadvisor"}'}

# Send an HTTP POST request to delete data series on the server
response = requests.post(url, params=params)

if not os.path.exists(checkpoint_path):
    os.makedirs(checkpoint_path)

pattern_file = "round-*-weights.npz"
files = [file for file in os.listdir(checkpoint_path) if fnmatch.fnmatch(file, pattern_file)]

if files:
    # Extract the round number from each file
    round_numbers = [int(re.search(r"round-(\d+)-weights\.npz", file).group(1)) for file in files]

    # Find the file with the highest round number
    latest_file = max(files, key=lambda file: int(re.search(r"round-(\d+)-weights\.npz", file).group(1)))

    weights = np.load(f"{checkpoint_path}/{latest_file}")
    parameters = fl.common.ndarrays_to_parameters([weights[key] for key in weights.files])

    last_round = max(round_numbers)

    strategy.initial_parameters = parameters
    strategy.set_round_offset(last_round)

# Start the federated server instance
server = fl.server.start_server(
    config=fl.server.ServerConfig(
        num_rounds=rounds), strategy=strategy)