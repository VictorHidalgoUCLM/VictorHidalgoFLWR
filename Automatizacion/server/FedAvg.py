from flwr.server.strategy import FedAvg
from config import config_path, prometheus_url, image, sleep_time
import sys
import os
import threading
import ast
import copy
import numpy as np
import flwr as fl
import time

from logging import WARNING
from typing import Callable, Dict, List, Optional, Tuple, Union
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

from data_analyst import DataAnalyst
import configparser
import pandas as pd

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
            

class FedAvgCustom(FedAvg):
    def __init__(self, num_exec, strategy_name, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.num_exec = num_exec
        self.strategy_name = strategy_name
        self.round_offset = 0
        self.config = configparser.ConfigParser()

        ruta_config = os.environ.get('CONFIG_PATH')
        self.config.read(ruta_config)

        self.configDevices = self.config.items('configDevices')
        listDispositivos = list(dict(self.configDevices))

        data = {
            'Global_accuracy': [],
            'Global_loss': [],
            'Local_accuracy': [],
            'Local_loss': [],
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
        # Obtener listas de epochs, batches y subsets desde la configuración
        epochs_list = ast.literal_eval(self.config.get('configClient', 'epochs'))
        batches_list = ast.literal_eval(self.config.get('configClient', 'batch_size'))
        subsets_list = ast.literal_eval(self.config.get('configClient', 'subset_size'))

        return epochs_list, batches_list, subsets_list
    
    def configure_fit(
        self, server_round: int, parameters: Parameters, client_manager: ClientManager
    ) -> List[Tuple[ClientProxy, FitIns]]:
        # Get client/config standard pairs from the FedAvg superclass
        client_config_pairs = super().configure_fit(
            server_round, parameters, client_manager
        )

        # Sort client/config pairs by client ID
        sorted_client_config_pairs = sorted(
            client_config_pairs, key=lambda x: x[0].cid
        )

        # Get lists of epochs, batches, and subsets from the configuration
        epochs_list, batches_list, subsets_list = self.getConfig()

        # Specific fit configuration for each client
        fit_conf = [
            {
                'epochs': int(epochs_list[i]),
                'batch_size': int(batches_list[i]),
                'subset_size': int(subsets_list[i]),
                'evaluate_on_fit': bool(False),
                'server_round': int(server_round),
            }
            for i in range(len(epochs_list))
        ]

        # Initialize data analyst if it is the first round
        if server_round == 1:
            analyst = DataAnalyst(config_path, prometheus_url, image, self.num_exec, self.strategy_name)

            # Get hostnames and create queries
            analyst.get_hostnames()
            analyst.create_queries()

            # Execute one-time queries
            analyst.execute_one_time_queries()

            # Start data export thread
            export_thread = ExportThread(analyst, sleep_time)
            export_thread.daemon = True
            export_thread.start()

        # Build list of fit configurations for each client
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
        
        directory_name = os.path.expanduser(self.config.get('configPaths', 'checkpoint').format(strategy=self.strategy_name, num_exec=self.config.getint('configVariable', 'num_exec')))

        if not os.path.exists(directory_name):
            os.makedirs(directory_name)

        for client_proxy, fit_res in results:
            for name, values in self.configDevices:
                try:
                    client_proxy.cid.index(eval(values)[0])
                    self.df_fit.loc[0, [name + '_fit_accuracy', name + '_fit_loss']] = [fit_res.metrics['accuracy'], fit_res.metrics['loss_distributed']]

                except ValueError:
                    pass

        if aggregated_parameters is not None:
            # Convert `Parameters` to `List[np.ndarray]`
            aggregated_ndarrays: List[np.ndarray] = fl.common.parameters_to_ndarrays(aggregated_parameters)

            # Save aggregated_ndarrays every 5 rounds
            if server_round % 5 == 0:
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

        directory_name = os.path.expanduser(self.config.get('configPaths', 'logs').format(strategy=self.strategy_name))

        if not os.path.exists(directory_name):
            os.makedirs(directory_name)

        for client_proxy, evaluate_res in results:
            for name, values in self.configDevices:
                try:
                    client_proxy.cid.index(eval(values)[0])
                    self.df_fit.loc[0, [name + '_ev_accuracy', name + '_ev_loss']] = [evaluate_res.metrics['accuracy'], evaluate_res.loss]

                except ValueError:
                    pass

        if loss_aggregated is not None and metrics_aggregated is not None:
            self.df_fit.loc[0, ['Global_accuracy', 'Global_loss']] = [metrics_aggregated['accuracy'], loss_aggregated]

            if not os.path.exists(f"{directory_name}/log_{self.num_exec}.csv"):
                self.df_fit.to_csv(f'{directory_name}/log_{self.num_exec}.csv', index=False, header=True)
            else:
                self.df_fit.to_csv(f'{directory_name}/log_{self.num_exec}.csv', index=False, header=False, mode='a')

        return loss_aggregated, metrics_aggregated