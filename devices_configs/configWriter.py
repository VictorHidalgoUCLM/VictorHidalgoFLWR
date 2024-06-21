import configparser

# Crear el objeto ConfigParser
config = configparser.ConfigParser()

# Configurar las secciones y los valores
config['configPaths'] = {
    'checkpoint': '~/Escritorio/results/{strategy}/checkpoints/{num_exec}',
    'metrics': '~/Escritorio/results/{strategy}/metrics',
    'logs': '~/Escritorio/results/{strategy}/logs',
    'server': '~/Escritorio/Automatizacion/server/server.py',
    'exception': '~/Escritorio/exception.txt'
}

config['configFixed'] = {
    'fraction_fit': '1',
    'fraction_evaluate': '1',
    'min_fit_clients': '3',
    'min_evaluate_clients': '3',
    'min_available_clients': '3',
    'evaluate_fn': 'None',
    'on_fit_config_fn': 'None',
    'on_evaluate_config_fn': 'None',
    'accept_failures': 'True',
    'initial_parameters': 'server_side',
    'fit_metrics_aggregation_fn': 'fit_weighted_average',
    'evaluate_metrics_aggregation_fn': 'evaluate_weighted_average',
    'rounds': '60'
}

config['configClient'] = {
    'epochs': "['1', '1', '1']",
    'batch_size': "['32', '32', '32']",
    'subset_size': "['1024', '1024', '256']"
}

config['configDevices'] = {
    'raspberrypi1': "['172.24.100.98', 'ES', 'RP1', 'victor']",
    'raspberry4': "['172.24.100.10', 'ES', 'RP4', 'victor']",
    'raspberry3': "['172.24.100.105', 'ES', 'RP3', 'victor']"
}

config['configFedProx'] = {
    'proximal_mu': '0.1'
}

config['configQFedAvg'] = {
    'q_param': '0.2',
    'qffl_learning_rate': '0.1'
}

config['configFedYogi'] = {
    'eta': '1e-2',
    'eta_l': '0.0316',
    'beta_1': '0.9',
    'beta_2': '0.99',
    'tau': '1e-3'
}


# Escribir el archivo de configuración
with open('config.ini', 'w') as configfile:
    config.write(configfile)

print("Archivo config.ini generado con éxito.")
