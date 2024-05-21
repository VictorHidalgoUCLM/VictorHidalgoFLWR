import requests
import yaml
import os
import time
import csv
import configparser

# Reads config_file from CONFIG_PATH
ruta_config = os.environ.get('CONFIG_PATH')
config_file = configparser.ConfigParser()
config_file.read(ruta_config)

# Get num_exec, strategy_name and ruta_metrics from config file
num_exec = config_file.get('configVariable', 'num_exec')
strategy_name = config_file.get('configVariable', 'strategy')

ruta_metrics = os.path.expanduser(config_file.get('configPaths', 'metrics').format(strategy=strategy_name))

# Class DataAnalyst
class DataAnalyst:
    """
    This class defines functions for measuring and storing metrics locally executing
    recursive queries.
    """

    def __init__(
            self,
            prometheus_config_path,
            prometheus_url,
            image,
            num_exec,
            config):

        # Other variables
        self.init_time = 0
        self.prometheus_config_path = prometheus_config_path
        self.prometheus_url = prometheus_url
        self.image = image
        self.num_exec = num_exec
        self.config = config

        # Variables for time and queries
        self.elapsed_time = 0
        self.prometheus_api_url = f"{self.prometheus_url}/api/v1/query"
        self.raspberry_pis = []
        self.output_files = []
        self.queries_one_time = []
        self.queries_recursive = []
        self.result_one_time = []
        self.result_recursive = []

        # Several counters for error detection
        self.same_cpu_counter = []
        self.prev_values = []


    def get_hostnames(self):
        """
        Function that searches for client hostnames defined at prometheus
        YAML config file through scraping.
        """
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

                                    # Append to raspberry_pis list
                                    self.raspberry_pis.append(hostname)

        # Init counter values
        self.same_cpu_counter = [0] * len(self.raspberry_pis)
        self.prev_values = [0] * len(self.raspberry_pis)


    def create_queries(self):
        """
        Create_queries defines recursive_queries and one_time_queries for each hostname.
        It saves the complete list of queries that later will be made.
        """
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
        """
        This function executes several queries and returns its result
        """
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
        """
        Function that executes all one_time_queries and waits until
        all clients have answered.
        """
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
        """
        Function that executes recursive queries (only one time)
        """

        self.elapsed_time = int(time.time() - self.init_time)
        self.result_recursive = []

        for query in self.queries_recursive:
            query_result = self.execute_query(query)

            query_values = [value['value'][1]
                            for result in query_result for value in result]
            self.result_recursive.append(query_values)

    def export_data(self):
        """
        Main function of DataAnalyst class. This function determines each client
        number of CPUs, RAM and Swap memory, and writes all data formatted at
        output_file, defined earlier in config file.
        """
        for i, _ in enumerate(self.raspberry_pis):
            output_file = self.output_files[i]

            cpu_total = self.result_one_time[i][0]
            memory_total = self.result_one_time[i][1]
            swap_total = self.result_one_time[i][2]

            result = self.result_recursive[i]
            
            # Writes header row if file is new and the data row obtained
            # as a result
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

            # Checks if any client is about to fail
            if self.prev_values[i] == row[3]:
                self.same_cpu_counter[i] += 1

                if self.same_cpu_counter[i] >= 15:
                    print(f"Va a fallar {self.raspberry_pis[i]}, tiempo {self.elapsed_time}.")

            else:
                self.prev_values[i] = row[3]
                self.same_cpu_counter[i] = 0