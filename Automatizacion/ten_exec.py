import time
import subprocess
import os
import configparser

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
    'rounds': '100',

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
    'epochs': ["1", "3", "3"],
    'batch_size': ["32", "32", "32"],
    'subset_size': ["2500", "2500", "2500"],
}

# Ruta al archivo de configuración
ruta_config = os.path.join(
    os.path.expanduser('~'),
    'Escritorio',
    'Automatizacion',
    'config.py')

# Crear y leer el archivo de configuración
config = configparser.ConfigParser()
config.read(ruta_config)

# Actualizar las secciones del archivo de configuración con las
# configuraciones definidas
config['ConfigFija'] = {**configFija}
config['ConfigCliente'] = {**configCliente}

# Escribir las configuraciones en el archivo de configuración
with open(ruta_config, 'w') as configfile:
    config.write(configfile)

# Ruta al script que se ejecutará
script_path = os.path.join(
    os.path.expanduser('~'),
    'Escritorio',
    'Automatizacion',
    'run.py')

# Número de ejecuciones y estrategias a probar
num_ejecuciones = 1
strategies = ["FedAvg", "FedProx", "QFedAvg", "FedYogi"]

# Bucle para ejecutar el script con diferentes configuraciones
for strategy in strategies:
    for i in range(num_ejecuciones):
        print(f"Ejecucion de estrategia {strategy}: {i+1}...")

        # Actualizar la sección 'ConfigVariable' con el número de ejecución y
        # la estrategia
        config['ConfigVariable'] = {
            'num_exec': f"{i+1}",
            'strategy': strategy,
            }

        # Escribir las configuraciones actualizadas en el archivo de
        # configuración
        with open(ruta_config, 'w') as configfile:
            config.write(configfile)

        # Comando para ejecutar el script
        command = ["python", script_path]

        # Iniciar el proceso hijo
        process = subprocess.Popen(command)
        process.wait()  # Esperar a que el proceso hijo termine

        print(f"Ejecucion {i+1} completada, cargando siguiente...")

        # Esperar 5 segundos entre ejecuciones
        time.sleep(5)
