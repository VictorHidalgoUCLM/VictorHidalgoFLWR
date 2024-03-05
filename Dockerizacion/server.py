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
from FedYogi import FedYogiCustom

# Ruta al archivo de configuración
ruta_config = os.path.expanduser('~/Escritorio/Automatizacion/config.ini')
config = configparser.ConfigParser()
config.read(ruta_config)

# Configuración de variables desde el archivo de configuración
num_exec = config.get('ConfigVariable', 'num_exec')
strategy_name = config.get('ConfigVariable', 'strategy')
rounds = config.getint('ConfigVariable', 'step_rounds')

ruta_checkpoints = os.path.expanduser(config.get('ConfigEntorno', 'checkpoint').format(strategy=strategy_name, num_exec=config.getint('ConfigVariable', 'num_exec')))

model = tf.keras.applications.MobileNet((32, 32, 1), classes=10, weights=None)
model.compile("adam", "sparse_categorical_crossentropy", metrics=["accuracy"])

# Función para calcular la métrica de promedio ponderado
def fit_weighted_average(
        metrics: List[Tuple[int, fl.common.Metrics]]) -> fl.common.Metrics:

    accuracies = [num_examples * m["accuracy"] for num_examples, m in metrics]
    losses = [num_examples * m["loss"] for num_examples, m in metrics]
    losses_distributed = [num_examples * m["loss_distributed"] for num_examples, m in metrics]

    examples = [num_examples for num_examples, _ in metrics]

    return {"accuracy": sum(accuracies) / sum(examples), "loss": sum(losses) / sum(examples), "loss_distributed": sum(losses_distributed) / sum(examples)}

# Función para calcular la métrica de promedio ponderado
def evaluate_weighted_average(
        metrics: List[Tuple[int, fl.common.Metrics]]) -> fl.common.Metrics:

    accuracies = [num_examples * m["accuracy"] for num_examples, m in metrics]

    examples = [num_examples for num_examples, _ in metrics]

    return {"accuracy": sum(accuracies) / sum(examples)}

# Diccionario de funciones que mapean nombres a funciones
funciones_dict = {
    'server_side': ndarrays_to_parameters(model.get_weights()),
    'fit_weighted_average': fit_weighted_average,
    'evaluate_weighted_average': evaluate_weighted_average,
    'None': None,
}

# Configuraciones extraídas del archivo de configuración
configuraciones = {
    'fraction_fit': config.getfloat('ConfigFija', 'fraction_fit'),
    'fraction_evaluate': config.getfloat('ConfigFija', 'fraction_evaluate'),
    'min_fit_clients': config.getint('ConfigFija', 'min_fit_clients'),
    'min_evaluate_clients': config.getint('ConfigFija', 'min_evaluate_clients'),
    'min_available_clients': config.getint('ConfigFija', 'min_available_clients'),
    'evaluate_fn': config.get('ConfigFija', 'evaluate_fn'),
    'on_fit_config_fn': config.get('ConfigFija', 'on_fit_config_fn'),
    'on_evaluate_config_fn': config.get('ConfigFija', 'on_evaluate_config_fn'),
    'accept_failures': config.getboolean('ConfigFija', 'accept_failures'),
    'initial_parameters': config.get('ConfigFija', 'initial_parameters'),
    'fit_metrics_aggregation_fn': config.get('ConfigFija', 'fit_metrics_aggregation_fn'),
    'evaluate_metrics_aggregation_fn': config.get('ConfigFija', 'evaluate_metrics_aggregation_fn'),
    'num_exec': num_exec,
    'strategy_name': strategy_name,
}

# Mapea valores de configuración a funciones según el diccionario
for key, value in configuraciones.items():
    if value in funciones_dict:
        configuraciones[key] = funciones_dict[value]

# Crear estrategia según la estrategia seleccionada en el archivo de
# configuración
def fedAvg():
    strategy = FedAvgCustom(**configuraciones)
    return strategy


def fedProx():
    strategy = FedProxCustom(**configuraciones,
                            proximal_mu=config.getfloat('ConfigFija', 'proximal_mu'),
                            )
    return strategy


def qFedAvg():
    strategy = QFedAvgCustom(**configuraciones,
                            q_param=config.getfloat('ConfigFija', 'q_param'),
                            qffl_learning_rate=config.getfloat('ConfigFija', 'qffl_learning_rate'),
                            )
    return strategy

def fedYogi():
    strategy = FedYogiCustom(**configuraciones,
                            eta=config.getfloat('ConfigFija', 'eta'),
                            eta_l=config.getfloat('ConfigFija', 'eta_l'),
                            beta_1=config.getfloat('ConfigFija', 'beta_1'),
                            beta_2=config.getfloat('ConfigFija', 'beta_2'),
                            tau=config.getfloat('ConfigFija', 'tau'),
                            )
    return strategy


# Función predeterminada si la estrategia no está seleccionada correctamente
def default():
    return "No se ha seleccionado nada"


# Mapea nombres de estrategias a funciones
switch = {
    "FedAvg": fedAvg,
    "FedProx": fedProx,
    "QFedAvg": qFedAvg,
    "FedYogi": fedYogi,
}

# Selecciona la estrategia según el nombre proporcionado en el archivo de
# configuración
selected_strategy = switch.get(strategy_name, default)
strategy = selected_strategy()

# Configuración para eliminar series de datos en el servidor
url = 'http://localhost:9090/api/v1/admin/tsdb/delete_series'
params = {'match[]': '{job="cadvisor"}'}

# Envía una solicitud HTTP POST para eliminar series de datos en el servidor
response = requests.post(url, params=params)

if not os.path.exists(ruta_checkpoints):
    os.makedirs(ruta_checkpoints)

patron_archivo = "round-*-weights.npz"
archivos = [archivo for archivo in os.listdir(ruta_checkpoints) if fnmatch.fnmatch(archivo, patron_archivo)]

if archivos:
    # Extraer el número de ronda de cada archivo
    numeros_de_ronda = [int(re.search(r"round-(\d+)-weights\.npz", archivo).group(1)) for archivo in archivos]

    # Encontrar el archivo con el número de ronda más alto
    archivo_mas_reciente = max(archivos, key=lambda archivo: int(re.search(r"round-(\d+)-weights\.npz", archivo).group(1)))

    weights = np.load(f"{ruta_checkpoints}/{archivo_mas_reciente}")
    parameters = fl.common.ndarrays_to_parameters(weights)

    last_round = max(numeros_de_ronda)

    strategy.initialize_parameters(parameters)
    strategy.set_round_offset(last_round)

# Inicia la instancia del servidor federado
server = fl.server.start_server(
    config=fl.server.ServerConfig(
        num_rounds=rounds), strategy=strategy)