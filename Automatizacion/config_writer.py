from configparser import ConfigParser
import os

# Set sections
config = ConfigParser()

# Set Entorno variables
configEntorno = {
    "checkpoint": f"~/Escritorio/{{strategy}}/checkpoints/{{num_exec}}",
    "metrics": f"~/Escritorio/{{strategy}}/metrics",
    "logs": f"~/Escritorio/{{strategy}}/logs",
    "server": "~/Escritorio/Dockerizacion/server.py",
    "exception": "~/Escritorio/exception.txt",
}

# Configuración fija para el archivo de configuración
configFija = {
    'fraction_fit': "1",
    'fraction_evaluate': "1",
    'min_fit_clients': "3",
    'min_evaluate_clients': "3",
    'min_available_clients': "3",
    'evaluate_fn': "None",
    'on_fit_config_fn': "None",
    'on_evaluate_config_fn': "None",
    'accept_failures': "True",
    'initial_parameters': "server_side",
    'fit_metrics_aggregation_fn': 'fit_weighted_average',
    'evaluate_metrics_aggregation_fn': 'evaluate_weighted_average',
    'rounds': '30',

    'proximal_mu': '0.1',

    'q_param': '0.2',
    'qffl_learning_rate': '0.1',

    'eta': '1e-2',
    'eta_l': '0.0316',
    'beta_1': '0.9',
    'beta_2': '0.99',
    'tau': '1e-3',
}

# Configuración específica para el cliente en el archivo de configuración
configCliente = {
    'epochs': ["3", "3", "3"],
    'batch_size': ["32", "32", "32"],
    'subset_size': ["2000", "2000", "2000"],
}

configDispositivos = {
    'raspberrypi1': '172.24.100.98',
    'raspberry5': '172.24.100.121',
    'raspberry6': '172.24.100.11',
}

config['ConfigEntorno'] = {**configEntorno}
config['ConfigFija'] = {**configFija}
config['ConfigCliente'] = {**configCliente}
config['ConfigDispositivos'] = {**configDispositivos}

# Write config at config.ini
config_file_path = os.path.expanduser("~/Escritorio/Automatizacion/config.ini")
with open(config_file_path, "w") as config_file:
    config.write(config_file)

print(f"Archivo de configuración '{config_file_path}' escrito...")