import requests
import yaml
import os
import time
import csv
import configparser

# Ruta al archivo de configuración
ruta_config = os.path.expanduser('~/Escritorio/Automatizacion/config.ini')
config_file = configparser.ConfigParser()
config_file.read(ruta_config)

# Obtener configuraciones del archivo
num_exec = config_file.get('ConfigVariable', 'num_exec')
strategy_name = config_file.get('ConfigVariable', 'strategy')

ruta_metrics = os.path.expanduser(config_file.get('ConfigEntorno', 'metrics').format(strategy=strategy_name))
ruta_excepcion = os.path.expanduser(config_file.get('ConfigEntorno', 'exception'))
ruta_flag = os.path.expanduser("~/Escritorio/exceptionFlag.txt")

class DataAnalyst:
    def __init__(
            self,
            prometheus_config_path,
            prometheus_url,
            image,
            num_exec,
            config):
        # Inicialización de variables
        self.init_time = 0
        self.prometheus_config_path = prometheus_config_path
        self.prometheus_url = prometheus_url
        self.image = image
        self.num_exec = num_exec
        self.config = config

        # Variables para el tiempo y las consultas
        self.elapsed_time = 0
        self.prometheus_api_url = f"{self.prometheus_url}/api/v1/query"
        self.raspberry_pis = []
        self.output_files = []
        self.queries_one_time = []
        self.queries_recursive = []
        self.result_one_time = []
        self.result_recursive = []


    def get_hostnames(self):
        # Obtener nombres de host desde el archivo de configuración de
        # Prometheus
        with open(self.prometheus_config_path, 'r') as config_file:
            config_data = yaml.safe_load(config_file)

            if 'scrape_configs' in config_data:
                for scrape_config in config_data['scrape_configs']:
                    if 'static_configs' in scrape_config:
                        for static_config in scrape_config['static_configs']:
                            if 'labels' in static_config:
                                hostname = static_config['labels'].get(
                                    'hostname')
                                if hostname:
                                    self.raspberry_pis.append(hostname)

    def create_queries(self):
        # Crear consultas para una vez y consultas recursivas
        recursive_queries = [
            'container_network_transmit_bytes_total{{hostname="{pi}",image="{image}"}}',
            'container_network_receive_bytes_total{{hostname="{pi}",image="{image}"}}',
            'container_cpu_usage_seconds_total{{hostname="{pi}",image="{image}"}}',
            'container_memory_usage_bytes{{hostname="{pi}",image="{image}"}}',
            'node_memory_SwapFree_bytes{{hostname="{pi}"}}'
            ]

        one_time_queries = [
            'machine_cpu_cores{{hostname="{pi}"}}',
            'machine_memory_bytes{{hostname="{pi}"}}',
            'node_memory_SwapFree_bytes{{hostname="{pi}"}}',
            'container_network_transmit_bytes_total{{hostname="{pi}",image="{image}"}}',
            'container_network_receive_bytes_total{{hostname="{pi}",image="{image}"}}',
            'container_cpu_usage_seconds_total{{hostname="{pi}",image="{image}"}}',
            'container_memory_usage_bytes{{hostname="{pi}",image="{image}"}}',
            'node_memory_SwapTotal_bytes{{hostname="{pi}"}}']

        for raspberry_pi in self.raspberry_pis:
            # Crear directorios para almacenar resultados
            if not os.path.exists(os.path.expanduser(
                    f'{ruta_metrics}/{raspberry_pi}')):
                os.makedirs(os.path.expanduser(
                    f'{ruta_metrics}/{raspberry_pi}'))

            output_file = f'{ruta_metrics}/{raspberry_pi}/ex_{num_exec}.csv'
            self.output_files.append(os.path.expanduser(output_file))

            # Crear consultas para una vez y consultas recursivas para cada
            # Raspberry Pi
            queries = [query.format(pi=raspberry_pi, image=self.image)
                       for query in recursive_queries]
            self.queries_recursive.append(queries)

            queries = [
                query.format(
                    pi=raspberry_pi,
                    image=self.image) for query in one_time_queries]
            self.queries_one_time.append(queries)

    def execute_query(self, queries):
        # Ejecutar consultas y obtener resultados
        result = []

        for query in queries:
            response = requests.get(
                self.prometheus_api_url, params={
                    'query': query})

            if response.status_code == 200:
                result.append(response.json()['data']['result'])
            else:
                print(
                    f"Error al realizar la consulta. Código de estado: {response.status_code}")
                return

        return result

    def execute_one_time_queries(self):
        # Ejecutar consultas de una vez y manejar resultados
        for query in self.queries_one_time:
            query_result = self.execute_query(query)

            while any(not res for res in query_result):
                query_result = self.execute_query(query)
                print("No se recibe, esperando...")
                time.sleep(5)

            print("Recibido")

            query_values = [result[0]['value'][1] for result in query_result]
            self.result_one_time.append(query_values)

        self.init_time = time.time()

    def execute_recursive_queries(self):
        # Ejecutar consultas recursivas y manejar resultados
        self.elapsed_time = int(time.time() - self.init_time)
        self.result_recursive = []

        for query in self.queries_recursive:
            query_result = self.execute_query(query)

            query_values = [value['value'][1]
                            for result in query_result for value in result]
            self.result_recursive.append(query_values)

    def export_data(self):
        # Exportar datos a archivos CSV
        for i, _ in enumerate(self.raspberry_pis):
            output_file = self.output_files[i]

            cpu_total = self.result_one_time[i][0]
            memory_total = self.result_one_time[i][1]
            swap_total = self.result_one_time[i][2]

            result = self.result_recursive[i]
            
            try:
                with open(output_file, 'a', newline='') as csvfile:
                    csv_writer = csv.writer(csvfile)

                    if os.path.getsize(output_file) == 0:
                        header_row = [
                            'Time_stamp(s)',
                            'Net_transmit(B)',
                            'Net_receive(B)',
                            'CPU_time(s)',
                            'RAM_usage(B)',
                            'Swap_free(B)',
                            '',
                            int(cpu_total),
                            int(memory_total),
                            int(swap_total)]
                        
                        csv_writer.writerow(header_row)

                    row = [self.elapsed_time] + [float(value) for value in result[:5]]
                    csv_writer.writerow(row)
                
            except IndexError as exception:
                with open(ruta_excepcion, 'w') as archivo:
                    archivo.write(str(exception))

                with open(ruta_flag, 'w') as archivo:
                    archivo.write(f"Hemos parado en la ejecución {num_exec}, de {strategy_name}, en el tiempo {self.elapsed_time}.")

                print("Error en data_analyst")