# Copyright 2020 Flower Labs GmbH. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""FAIR RESOURCE ALLOCATION IN FEDERATED LEARNING [Li et al., 2020] strategy.

Paper: openreview.net/pdf?id=ByexElSYDr
"""

from itertools import count
from logging import WARNING
from typing import Callable, Dict, List, Optional, Tuple, Union

import ast
import numpy as np
from  flwr.server.server import Server
from flwr.common import (
    EvaluateIns,
    EvaluateRes,
    FitIns,
    FitRes,
    MetricsAggregationFn,
    NDArrays,
    Parameters,
    Scalar,
    ndarrays_to_parameters,
    parameters_to_ndarrays,
)
from flwr.common.logger import log
from flwr.server.client_manager import ClientManager
from flwr.server.client_proxy import ClientProxy

from flwr.server.strategy.aggregate import aggregate_qffl, weighted_loss_avg
from flwr.server.strategy.fedyogi import FedYogi

from config import config_path, prometheus_url, image, sleep_time
import threading
import sys
import os
sys.path.insert(1, os.path.expanduser("~/Escritorio/monitorizacion"))
from data_analyst import DataAnalyst
import configparser
import time
import pandas as pd
import flwr as fl
import numpy as np

class ExportThread(threading.Thread):
    def __init__(self, analyst_instance, sleep_time):
        super().__init__()
        self.analyst_instance = analyst_instance
        self.sleep_time = sleep_time

    def run(self):
        try:
            while True:
                self.analyst_instance.execute_recursive_queries()
                self.analyst_instance.export_data()
                time.sleep(self.sleep_time)

        except IndexError as e:
            print("Error en Run")

# pylint: disable=too-many-locals
class FedYogiCustom(FedYogi):
    """Configurable QFedAvg strategy implementation."""

    # pylint: disable=too-many-arguments,too-many-instance-attributes
    def __init__(self, num_exec, strategy_name, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.num_exec = num_exec
        self.strategy_name = strategy_name
        self.round_offset = 0
        self.config = configparser.ConfigParser()

        ruta_config = os.path.expanduser('~/Escritorio/Automatizacion/config.ini')
        self.config.read(ruta_config)

        self.configDispositivos = self.config.items('ConfigDispositivos')
        listDispositivos = list(dict(self.configDispositivos))

        data = {
            'Global_accuracy': [],
            'Global_loss': [],
            'Local_accuracy': [],
            'Local_loss': []
        }

        for dispositivo in listDispositivos:
            data[dispositivo + '_fit_accuracy'] = []
            data[dispositivo + '_fit_loss'] = []
            data[dispositivo + '_ev_accuracy'] = []
            data[dispositivo + '_ev_loss'] = []

        self.df_fit = pd.DataFrame(data)

    def set_round_offset(self, offset):
        self.round_offset = offset

    def getConfig(self):
        epochs_list = ast.literal_eval(self.config.get('ConfigCliente', 'epochs'))
        batches_list = ast.literal_eval(self.config.get('ConfigCliente', 'batch_size'))
        subsets_list = ast.literal_eval(self.config.get('ConfigCliente', 'subset_size'))

        return epochs_list, batches_list, subsets_list

    def configure_fit(
        self, server_round: int, parameters: Parameters, client_manager: ClientManager
    ) -> List[Tuple[ClientProxy, FitIns]]:

        """Configure the next round of training."""
        config = {}
        if self.on_fit_config_fn is not None:
            # Custom fit config function provided
            config = self.on_fit_config_fn(server_round)
        fit_ins = FitIns(parameters, config)

        # Sample clients
        sample_size, min_num_clients = self.num_fit_clients(
            client_manager.num_available()
        )
        clients = client_manager.sample(
            num_clients=sample_size, min_num_clients=min_num_clients
        )

        client_config_pairs = [(client, fit_ins) for client in clients]

        # Ordenar los pares cliente/configuración por ID de cliente
        sorted_client_config_pairs = sorted(
            client_config_pairs, key=lambda x: x[0].cid
        )

        epochs_list, batches_list, subsets_list = self.getConfig()

        fit_conf = [
            {
                'epochs': int(epochs_list[i]),
                'batch_size': int(batches_list[i]),
                'subset_size': int(subsets_list[i]),
                'evaluate_on_fit': bool(False),
            }
            for i in range(len(epochs_list))
        ]

        # Inicializar análisis de datos si es la primera ronda
        if server_round == 1:
            analyst = DataAnalyst(config_path, prometheus_url, image, self.num_exec, self.strategy_name)

            # Obtener nombres de host y crear consultas
            analyst.get_hostnames()
            analyst.create_queries()

            # Ejecutar consultas de una vez
            analyst.execute_one_time_queries()

            # Iniciar hilo de exportación de datos
            export_thread = ExportThread(analyst, sleep_time)
            export_thread.daemon = True
            export_thread.start()


        return [
            (
                client,
                FitIns(
                    fit_ins.parameters,
                    {**fit_conf[i]},
                ),
            )
            for (client, fit_ins), i in zip(sorted_client_config_pairs, range(len(sorted_client_config_pairs)))
        ]
    

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[fl.server.client_proxy.ClientProxy, fl.common.FitRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:

        # Call aggregate_fit from base class (FedAvg) to aggregate parameters and metrics
        aggregated_parameters, aggregated_metrics = super().aggregate_fit(server_round, results, failures)
        
        directory_name = os.path.expanduser(self.config.get('ConfigEntorno', 'checkpoint').format(strategy=self.strategy_name))

        if not os.path.exists(directory_name):
            os.makedirs(directory_name)

        for client_proxy, fit_res in results:
            for nombre, valor in self.configDispositivos:
                try:
                    client_proxy.cid.index(valor)
                    self.df_fit.loc[0, [nombre + '_fit_accuracy', nombre + '_fit_loss']] = [fit_res.metrics['accuracy'], fit_res.metrics['loss_distributed']]

                except ValueError:
                    pass

        if aggregated_parameters is not None:
            # Convert `Parameters` to `List[np.ndarray]`
            aggregated_ndarrays: List[np.ndarray] = fl.common.parameters_to_ndarrays(aggregated_parameters)

            # Save aggregated_ndarrays
            print(f"Saving round {server_round} aggregated_ndarrays...")
            np.savez(f"{directory_name}/round-{server_round+self.round_offset}-weights.npz", *aggregated_ndarrays)

            self.df_fit.loc[0, ['Local_accuracy', 'Local_loss']] = [aggregated_metrics['accuracy'], aggregated_metrics['loss_distributed']]

        return aggregated_parameters, aggregated_metrics
    

    def aggregate_evaluate(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, EvaluateRes]],
        failures: List[Union[Tuple[ClientProxy, EvaluateRes], BaseException]],
    ) -> Tuple[Optional[float], Dict[str, Scalar]]:
        
        loss_aggregated, metrics_aggregated = super().aggregate_evaluate(server_round, results, failures)

        directory_name = os.path.expanduser(self.config.get('ConfigEntorno', 'logs').format(strategy=self.strategy_name))

        if not os.path.exists(directory_name):
            os.makedirs(directory_name)

        for client_proxy, evaluate_res in results:
            for nombre, valor in self.configDispositivos:
                try:
                    client_proxy.cid.index(valor)
                    self.df_fit.loc[0, [nombre + '_ev_accuracy', nombre + '_ev_loss']] = [evaluate_res.metrics['accuracy'], evaluate_res.loss]

                except ValueError:
                    pass

        if loss_aggregated is not None and metrics_aggregated is not None:
            self.df_fit.loc[0, ['Global_accuracy', 'Global_loss']] = [metrics_aggregated['accuracy'], loss_aggregated]

            if not os.path.exists(f"{directory_name}/log_{self.num_exec}.csv"):
                self.df_fit.to_csv(f'{directory_name}/log_{self.num_exec}.csv', index=False, header=True)
            else:
                self.df_fit.to_csv(f'{directory_name}/log_{self.num_exec}.csv', index=False, header=False, mode='a')

        return loss_aggregated, metrics_aggregated