from configparser import ConfigParser

# Set sections
config = ConfigParser()

configPaths = {
    "checkpoint": f"~/Escritorio/results/{{strategy}}/checkpoints/{{num_exec}}",
    "metrics": f"~/Escritorio/results/{{strategy}}/metrics",
    "logs": f"~/Escritorio/results/{{strategy}}/logs",
    "server": "~/Escritorio/Automatizacion/server/server.py",
    "exception": "~/Escritorio/exception.txt",
}

configFixed = {
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
}

configClient = {
    'epochs': ["3", "3", "12"],
    'batch_size': ["32", "32", "64"],
    'subset_size': ["1000", "1000", "3000"],
}

configDevices = {
    'raspberrypi1': ['172.24.100.98', 'ES', 'RP1', 'victor'],
    'raspberry4': ['172.24.100.10', 'ES', 'RP4', 'victor'],
    'raspberry3': ['172.24.100.105', 'ES', 'RP3', 'victor'],
}

# Individual configurations for strategies
configFedProx = {
    'proximal_mu': '0.1',
}

configQFedAvg = {
    'q_param': '0.2',
    'qffl_learning_rate': '0.1',
}

configFedYogi = {
    'eta': '1e-2',
    'eta_l': '0.0316',
    'beta_1': '0.9',
    'beta_2': '0.99',
    'tau': '1e-3',
}

config['configPaths'] = {**configPaths}
config['configFixed'] = {**configFixed}
config['configClient'] = {**configClient}
config['configDevices'] = {**configDevices}
config['configFedProx'] = {**configFedProx}
config['configQFedAvg'] = {**configQFedAvg}
config['configFedYogi'] = {**configFedYogi}

# Write config at config.ini
file_path = "config.ini"
with open(file_path, "w") as config_file:
    config.write(config_file)

print(f"Config file '{file_path}' written...")