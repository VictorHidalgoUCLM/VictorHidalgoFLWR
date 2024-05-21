# Flwr imports
from flwr.server.strategy import FedProx
from typing import Dict, List, Optional, Tuple, Union
from flwr.common import (
    EvaluateRes,
    FitIns,
    FitRes,
    Parameters,
    Scalar,
    parameters_to_ndarrays,
)
from flwr.server.client_manager import ClientManager
from flwr.server.client_proxy import ClientProxy

# Utility imports
from ExportThread import ExportThread
import os
import ast
import numpy as np
from data_analyst import DataAnalyst
import configparser
import pandas as pd

class FedProxCustom(FedProx):
    """
    Class that modifies the original behaviour of FedProx, allowing the server
    to configure client options (epochs, batch_size...) and query to prometheus
    from server, storing all data locally.

    Its init method receives num_exec and strategy_name, which will be used to
    store metrics and results.
    """

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

        self.sleep_time = self.config.getint('configPrometheus', 'sleep_time')

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
        """
        This function receives a number (offset) and stores it at self.round_offset
        """
        self.round_offset = offset

    def getConfig(self):
        """
        This function reads client configs and returns them.
        """

        epochs_list = ast.literal_eval(self.config.get('configClient', 'epochs'))
        batches_list = ast.literal_eval(self.config.get('configClient', 'batch_size'))
        subsets_list = ast.literal_eval(self.config.get('configClient', 'subset_size'))

        return epochs_list, batches_list, subsets_list

    def configure_fit(
        self, server_round: int, parameters: Parameters, client_manager: ClientManager
    ) -> List[Tuple[ClientProxy, FitIns]]:
        """
        This function adds epochs, batch_size and subset_size to client configuration on fit.
        Additionally, it starts a thread that will execute metric querying.
        """

        # Obtener pares cliente/configuración estándar de la superclase FedProx
        client_config_pairs = super().configure_fit(
            server_round, parameters, client_manager
        )

        # Ordenar los pares cliente/configuración por ID de cliente
        sorted_client_config_pairs = sorted(
            client_config_pairs, key=lambda x: x[0].cid
        )

        # Obtener listas de epochs, batches y subsets desde la configuración
        epochs_list, batches_list, subsets_list = self.getConfig()

        # Configuración específica de ajuste para cada cliente
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

        # Inicializar análisis de datos si es la primera ronda
        if server_round == 1:
            analyst = DataAnalyst(self.num_exec, self.strategy_name)

            # Obtener nombres de host y crear consultas
            analyst.get_hostnames()
            analyst.create_queries()

            # Ejecutar consultas de una vez
            analyst.execute_one_time_queries()

            # Iniciar hilo de exportación de datos
            export_thread = ExportThread(analyst, self.sleep_time)
            export_thread.daemon = True
            export_thread.start()


        # Construir lista de configuraciones de ajuste para cada cliente
        return [
            (
                client,
                FitIns(
                    fit_ins.parameters,
                    {**fit_conf[i], "proximal_mu": self.proximal_mu},
                ),
            )
            for (client, fit_ins), i in zip(sorted_client_config_pairs, range(len(sorted_client_config_pairs)))
        ]
    
    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        """
        This function receives aggregate results and aggregate them, if current
        round is multiple of 5, a checkpoint is saved on serverside.
        """

        # Call aggregate_fit from base class (FedAvg) to aggregate parameters and metrics
        aggregated_parameters, aggregated_metrics = super().aggregate_fit(server_round, results, failures)
        
        directory_name = os.path.expanduser(self.config.get('configPaths', 'checkpoint').format(strategy=self.strategy_name, num_exec=self.config.getint('configVariable', 'num_exec')))

        if not os.path.exists(directory_name):
            os.makedirs(directory_name)

        for client_proxy, fit_res in results:
            for nombre, valores in self.configDevices:
                try:
                    client_proxy.cid.index(eval(valores)[0])
                    self.df_fit.loc[0, [nombre + '_fit_accuracy', nombre + '_fit_loss']] = [fit_res.metrics['accuracy'], fit_res.metrics['loss_distributed']]

                except ValueError:
                    pass

        if aggregated_parameters is not None:
            # Convert `Parameters` to `List[np.ndarray]`
            aggregated_ndarrays: List[np.ndarray] = parameters_to_ndarrays(aggregated_parameters)

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
        """
        This function receives evaluate results from clients and aggregates the results.
        """
        
        loss_aggregated, metrics_aggregated = super().aggregate_evaluate(server_round, results, failures)

        directory_name = os.path.expanduser(self.config.get('configPaths', 'logs').format(strategy=self.strategy_name))

        if not os.path.exists(directory_name):
            os.makedirs(directory_name)

        for client_proxy, evaluate_res in results:
            for nombre, valores in self.configDevices:
                try:
                    client_proxy.cid.index(eval(valores)[0])
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