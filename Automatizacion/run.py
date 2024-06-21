"""
This code runs the server and remote clients in the Federated Learning (FL)
process, allowing the execution of multiple consecutive configurations. Executes
num_exec times the same configuration, written at "./config.ini" with "./config_writer.py"

Each different thread executes and terminates the server/client code, and checks
every 10 seconds if Ctrl-C has been detected, or if process has terminated normally or if
there has been a problem with the metrics.
"""

# Necessary modules
import threading
import subprocess
import time
from configparser import ConfigParser
import fnmatch
import re
import signal
import os
import pandas as pd

handler_flag = False

def signal_handler(sig, frame, event):
    """
    Receives the signal to terminate the program, activates the event to 
    terminate all active threads.
    """
    global handler_flag

    print("Ctrl-C detected. Terminating all threads...")
    event.set()     # This event will be read later on each thread
    handler_flag = True


def get_last_round(config):
    """
    Auxiliary function to know how many rounds are left to execute in case 
    the program terminated abruptly, returning the next round to execute.
    """

    strategy_name = config.get('configVariable', 'strategy')
    directory_name = os.path.expanduser(config.get('configPaths', 'checkpoint').format(strategy=strategy_name, num_exec=config.getint('configVariable', 'num_exec')))

    if not os.path.exists(directory_name):
        os.makedirs(directory_name)

    file_pattern = "round-*-weights.npz"
    files = [file for file in os.listdir(directory_name) if fnmatch.fnmatch(file, file_pattern)]

    if files:
        # Extraer el número de ronda de cada file
        round_numbers = [int(re.search(r"round-(\d+)-weights\.npz", file).group(1)) for file in files]
        return max(round_numbers)
    
    else:
        return 0


def code_local(file_name, event, config):
    """
    Contains the code to be executed on the FL server and its termination code.
    """

    print(f"Local thread running FL server...")

    # Code for starting and ending local server code
    server = os.path.expanduser(config.get('configPaths', 'server'))
    comm_exec = f'python {server}'
    comm_term = f'pkill -f "python {server}"'

    # Execution of server
    process = subprocess.Popen(comm_exec, shell=True, text=True)

    # Checks every 10 seconds if has to end
    while process.poll() is None and not os.path.exists(file_name) and not event.is_set():
        time.sleep(10)

    # Terminates process if it has to end
    subprocess.run(comm_term, shell=True, text=True) 
    print("Local thread terminating...")

    process.terminate()
    process.wait()


def code_thread(id, user, ip, file_name, event):
    """
    Contains the code to be executed on an FL client and its termination code.
    """

    print(f"Remote thread {id} running FL client...")

    comm_list = [
        "cd slave",
        "docker-compose down",
        "docker-compose up -d",
        "cd ..",
        "docker pull victorhidalgo/client",
        'docker images -f "dangling=true" -q | xargs docker rmi',
        'docker run --rm -v ~/data:/data -it victorhidalgo/client',
    ]

    # Definition of exec and term commands
    comm_exec = f'ssh {user}@{ip} "script -q -c \\"{" && ".join(comm_list)}\\" /dev/null"'
    comm_term = f'ssh {user}@{ip} "(sudo pkill python3)"'

    # Execution of client process
    process = subprocess.Popen(comm_exec, shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, text=True)

    # Checks if it has to end every 10 seconds
    while process.poll() is None and not os.path.exists(file_name) and not event.is_set():
        time.sleep(10)

    if os.path.exists(file_name):
        event.set()

    # If it has to end, waits for server stopping and client process ends
    time.sleep(15)
    subprocess.run(comm_term, stderr=subprocess.DEVNULL, stdout=subprocess.DEVNULL, shell=True, text=True)
    print(f"Remote thread {id} terminating...")

    process.terminate()
    process.wait()


def main():
    # Event variable and SIGINT handler
    global handler_flag

    event = threading.Event()
    signal.signal(signal.SIGINT, lambda sig, frame: signal_handler(sig, frame, event))

    # Path to config file config.ini
    path_config = os.environ.get('CONFIG_PATH')
    config = ConfigParser()
    config.read(path_config)
    
    # Reading config file
    rounds = config.getint('configFixed', 'rounds')
    devices = config.items('configDevices')
    path_exception = os.path.expanduser(config.get('configPaths', 'exception'))
    
    # Number of executions and list of strategies
    num_exec = 10
    strategies = ["FedAvg"]

    # Foor loop for each strategy
    for strategy in strategies:
        # For loop for each execution
        for act_exec in range(num_exec):
            # Set current for loop configuration at config file
            config['configVariable'] = {
                'num_exec': f"{act_exec+1}",
                'strategy': strategy,
            }

            path_logs = os.path.expanduser(config.get('configPaths', 'logs').format(strategy=strategy))

            with open(path_config, 'w') as configfile:
                config.write(configfile)

            # Round number init
            # Calculates if previously we did not complete all rounds
            step_rounds = rounds - get_last_round(config)

            # While rounds left to execute
            while step_rounds > 0:
                config.set('configVariable', 'step_rounds', f'{step_rounds}')
            
                with open(path_config, 'w') as configfile:
                    config.write(configfile)

                # List of threads
                threads = []

                # Start all remote threads and append to list of threads
                for i, (user, values) in enumerate(devices, start=1):
                    ip = eval(values)[0]
                    thread = threading.Thread(target=code_thread, args=(i, user, ip, path_exception, event,))
                    threads.append(thread)

                # Start local thread and append to list
                thread = threading.Thread(target=code_local, args=(path_exception, event, config,))
                threads.append(thread)

                # Start of threads
                for thread in threads:
                    thread.start()

                # Join all threads
                for thread in threads:
                    thread.join()

                # Exit if abrupt termination (SIGINT)
                if event.is_set():
                    print("All threads terminated, exiting...")
                    event.clear()

                    if handler_flag:
                        exit()
            
                # Recalculates left rounds
                step_rounds = rounds - get_last_round(config)

                if os.path.exists(path_exception):
                    os.remove(path_exception)

                    # Leer el archivo CSV
                    df = pd.read_csv(f'{path_logs}/log_{act_exec+1}.csv')

                    # Seleccionar las primeras 10 filas
                    df_first = df.head(get_last_round(config))

                    # Guardar las primeras 10 filas de vuelta al archivo CSV
                    df_first.to_csv(f'{path_logs}/log_{act_exec+1}.csv', index=False)
            
if __name__ == "__main__":
    main()