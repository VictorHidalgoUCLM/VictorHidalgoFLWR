from flwr.server.strategy import QFedAvg
from config import config_path, prometheus_url, image, sleep_time
import os
import threading
import ast
import numpy as np
import flwr as fl
import time

from typing import Dict, List, Optional, Tuple, Union
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
from flwr.server.strategy.aggregate import aggregate_qffl
from flwr.server.client_manager import ClientManager
from flwr.server.client_proxy import ClientProxy
from flwr.common.logger import log

from data_analyst import DataAnalyst
import configparser
import pandas as pd
from logging import WARNING


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
class QFedAvgCustom(QFedAvg):
    """Configurable QFedAvg strategy implementation."""

    # pylint: disable=too-many-arguments,too-many-instance-attributes
    def __init__(self, num_exec, strategy_name, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.num_exec = num_exec
        self.strategy_name = strategy_name
        self.pre_results = None

        self.round_offset = 0
        self.config = configparser.ConfigParser()

        ruta_config = os.environ.get('CONFIG_PATH')
        self.config.read(ruta_config)

        self.configDevices = self.config.items('configDevices')
        listDispositivos = list(dict(self.configDevices))

        data = {
            'Global_accuracy': [],
            'Global_loss': [],
            'Global_precision': [],
            'Global_recall': [],
            'Global_f1_score': [],
            'Local_accuracy': [],
            'Local_loss': [],
            'Local_precision': [],
            'Local_recall': [],
            'Local_f1_score': [],
        }

        for dispositivo in listDispositivos:
            data[dispositivo + '_fit_accuracy'] = []
            data[dispositivo + '_fit_loss'] = []
            data[dispositivo + '_fit_precision'] = []
            data[dispositivo + '_fit_recall'] = []
            data[dispositivo + '_fit_f1_score'] = []
            data[dispositivo + '_ev_accuracy'] = []
            data[dispositivo + '_ev_loss'] = []
            data[dispositivo + '_ev_precision'] = []
            data[dispositivo + '_ev_recall'] = []
            data[dispositivo + '_ev_f1_score'] = []

        self.df_fit = pd.DataFrame(data)

    def set_round_offset(self, offset):
        self.round_offset = offset

    def getConfig(self):
        epochs_list = ast.literal_eval(self.config.get('configClient', 'epochs'))
        batches_list = ast.literal_eval(self.config.get('configClient', 'batch_size'))
        subsets_list = ast.literal_eval(self.config.get('configClient', 'subset_size'))

        return epochs_list, batches_list, subsets_list

    def configure_fit(
        self, server_round: int, parameters: Parameters, client_manager: ClientManager
    ) -> List[Tuple[ClientProxy, FitIns]]:

        """Configure the next round of training."""
        # Obtener pares cliente/configuración estándar de la superclase FedAvg
        client_config_pairs = super().configure_fit(
            server_round, parameters, client_manager
        )

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
                'evaluate_on_fit': bool(True),
                'server_round': int(server_round),
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
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        """Aggregate fit results using weighted average."""
        if not results:
            return None, {}
        # Do not aggregate if there are failures and failures are not accepted
        if not self.accept_failures and failures:
            return None, {}
        # Convert results

        def norm_grad(grad_list: NDArrays) -> float:
            # input: nested gradients
            # output: square of the L-2 norm
            client_grads = grad_list[0]
            for i in range(1, len(grad_list)):
                client_grads = np.append(
                    client_grads, grad_list[i]
                )  # output a flattened array
            squared = np.square(client_grads)
            summed = np.sum(squared)
            return float(summed)

        deltas = []
        hs_ffl = []

        if self.pre_weights is None:
            raise AttributeError("QffedAvg pre_weights are None in aggregate_fit")

        weights_before = self.pre_weights
        loss_list = []

        if self.pre_results is not None:
            for _, fit_res in self.pre_results:
                loss_list.append(fit_res.metrics["loss"])

            loss = sum(loss_list) / len(self.pre_results)

        else:
            loss = 1.0
            
        self.pre_results = results

        for _, fit_res in results:
            new_weights = parameters_to_ndarrays(fit_res.parameters)
            # plug in the weight updates into the gradient
            grads = [
                np.multiply((u - v), 1.0 / self.learning_rate)
                for u, v in zip(weights_before, new_weights)
            ]
            deltas.append(
                [np.float_power(loss + 1e-10, self.q_param) * grad for grad in grads]
            )
            # estimation of the local Lipschitz constant
            hs_ffl.append(
                self.q_param
                * np.float_power(loss + 1e-10, (self.q_param - 1))
                * norm_grad(grads)
                + (1.0 / self.learning_rate)
                * np.float_power(loss + 1e-10, self.q_param)
            )

        weights_aggregated: NDArrays = aggregate_qffl(weights_before, deltas, hs_ffl)
        parameters_aggregated = ndarrays_to_parameters(weights_aggregated)

        # Aggregate custom metrics if aggregation fn was provided
        metrics_aggregated = {}
        if self.fit_metrics_aggregation_fn:
            fit_metrics = [(res.num_examples, res.metrics) for _, res in results]
            metrics_aggregated = self.fit_metrics_aggregation_fn(fit_metrics)
        elif server_round == 1:  # Only log this warning once
            log(WARNING, "No fit_metrics_aggregation_fn provided")

        aggregated_parameters = parameters_aggregated
        aggregated_metrics = metrics_aggregated

        directory_name = os.path.expanduser(self.config.get('configPaths', 'checkpoint').format(strategy=self.strategy_name, num_exec=self.config.getint('configVariable', 'num_exec')))

        if not os.path.exists(directory_name):
            os.makedirs(directory_name)

        for client_proxy, fit_res in results:
            for name, valores in self.configDevices:
                try:
                    client_proxy.cid.index(eval(valores)[0])
                    self.df_fit.loc[0, [name + '_fit_accuracy', name + '_fit_loss', name + '_fit_precision', name + '_fit_recall', name + '_fit_f1_score']] = [fit_res.metrics['accuracy'], fit_res.metrics['loss_distributed'], fit_res.metrics['precision'], fit_res.metrics['recall'], fit_res.metrics['f1_score']]

                except ValueError:
                    pass

        if aggregated_parameters is not None:
            # Convert `Parameters` to `List[np.ndarray]`
            aggregated_ndarrays: List[np.ndarray] = fl.common.parameters_to_ndarrays(aggregated_parameters)

            # Save aggregated_ndarrays every 5 rounds
            if server_round % 5 == 0:
                print(f"Saving round {server_round} aggregated_ndarrays...")
                np.savez(f"{directory_name}/round-{server_round+self.round_offset}-weights.npz", *aggregated_ndarrays)

            self.df_fit.loc[0, ['Local_accuracy', 'Local_loss', 'Local_precision', 'Local_recall', 'Local_f1_score']] = [aggregated_metrics['accuracy'], aggregated_metrics['loss_distributed'], aggregated_metrics['precision'], aggregated_metrics['recall'], aggregated_metrics['f1_score']]

        return parameters_aggregated, metrics_aggregated
    

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
            for name, valores in self.configDevices:
                try:
                    client_proxy.cid.index(eval(valores)[0])
                    self.df_fit.loc[0, [name + '_ev_accuracy', name + '_ev_loss', name + '_ev_precision', name + '_ev_recall', name + '_ev_f1_score']] = [evaluate_res.metrics['accuracy'], evaluate_res.loss, evaluate_res.metrics['precision'], evaluate_res.metrics['recall'], evaluate_res.metrics['f1_score']]

                except ValueError:
                    pass

        if loss_aggregated is not None and metrics_aggregated is not None:
            self.df_fit.loc[0, ['Global_accuracy', 'Global_loss', 'Global_precision', 'Global_recall', 'Global_f1_score']] = [metrics_aggregated['accuracy'], loss_aggregated, metrics_aggregated['precision'], metrics_aggregated['recall'], metrics_aggregated['f1_score']]

            if not os.path.exists(f"{directory_name}/log_{self.num_exec}.csv"):
                self.df_fit.to_csv(f'{directory_name}/log_{self.num_exec}.csv', index=False, header=True)
            else:
                self.df_fit.to_csv(f'{directory_name}/log_{self.num_exec}.csv', index=False, header=False, mode='a')

        return loss_aggregated, metrics_aggregated