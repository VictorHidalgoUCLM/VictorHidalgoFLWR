import threading
import subprocess
import time
from configparser import ConfigParser
import fnmatch
import re
import signal

import smtplib
from email.mime.text import MIMEText
import os

def signal_handler(sig, frame, event):
    print("Ctrl-C detectado. Terminando todos los hilos...")
    event.set()


def get_last_round(config):
    strategy_name = config.get('ConfigVariable', 'strategy')
    directory_name = os.path.expanduser(config.get('ConfigEntorno', 'checkpoint').format(strategy=strategy_name, num_exec=config.getint('ConfigVariable', 'num_exec')))

    if not os.path.exists(directory_name):
        os.makedirs(directory_name)

    patron_archivo = "round-*-weights.npz"
    archivos = [archivo for archivo in os.listdir(directory_name) if fnmatch.fnmatch(archivo, patron_archivo)]

    if archivos:
        # Extraer el número de ronda de cada archivo
        numeros_de_ronda = [int(re.search(r"round-(\d+)-weights\.npz", archivo).group(1)) for archivo in archivos]
        return max(numeros_de_ronda)
    
    else:
        return 0


def codigo_local(nombre_archivo, event, config):
    print(f"Hilo local ejecutando servidor...")

    server = os.path.expanduser(config.get('ConfigEntorno', 'server'))
    comando_exec = f'python {server}'
    comando_fin = f'pkill -f "python {server}"'

    proceso = subprocess.Popen(comando_exec, shell=True, text=True)

    while proceso.poll() is None and not os.path.exists(nombre_archivo) and not event.is_set():
        time.sleep(10)

    if os.path.exists(nombre_archivo):
        subprocess.run(comando_fin, shell=True, text=True)
    
    print("Hilo local terminando...")

    proceso.terminate()
    proceso.wait()


def codigo_hilo(id, usuario, ip, nombre_archivo, event):
    print(f"Hilo remoto {id} ejecutando cliente...")

    comando_list = [
        "cd slave",
        "docker-compose down",
        "docker-compose up -d",
        "cd ..",
        "docker pull victorhidalgo/client",
        "docker run --rm -v ~/data:/data -it victorhidalgo/client",
    ]

    comando_exec = f'ssh {usuario}@{ip} "script -q -c \\"{" && ".join(comando_list)}\\" /dev/null"'
    comando_fin = f'ssh {usuario}@{ip} "(sudo pkill python3)"'

    proceso = subprocess.Popen(comando_exec, shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, text=True)

    while proceso.poll() is None and not os.path.exists(nombre_archivo) and not event.is_set():
        time.sleep(10)

    if os.path.exists(nombre_archivo):
        subprocess.run(comando_fin, stderr=subprocess.DEVNULL, stdout=subprocess.DEVNULL, shell=True, text=True)

    print(f"Hilo remoto {id} terminando...")

    proceso.terminate()
    proceso.wait()


def main():
    event = threading.Event()
    signal.signal(signal.SIGINT, lambda sig, frame: signal_handler(sig, frame, event))

    # Ruta al archivo de configuración
    ruta_config = os.path.expanduser("~/Escritorio/Automatizacion/config.ini")
    config = ConfigParser()

    config.read(ruta_config)
    
    rounds = config.getint('ConfigFija', 'rounds')
    dispositivos = config.items('ConfigDispositivos')
    ruta_excepcion = os.path.expanduser(config.get('ConfigEntorno', 'exception'))
    
    # Número de ejecuciones y estrategias a probar
    num_ejecuciones = 10
    strategies = ["FedAvg"]

    for strategy in strategies:
        # Configuramos la ronda
        for num_exec in range(num_ejecuciones):
            config['ConfigVariable'] = {
                'num_exec': f"{num_exec+1}",
                'strategy': strategy,
            }

            with open(ruta_config, 'w') as configfile:
                config.write(configfile)

            # Inicializamos la ronda
            prev_rounds = get_last_round(config)
            step_rounds = rounds

            while step_rounds > 0:
                config.set('ConfigVariable', 'step_rounds', f'{step_rounds}')
            
                with open(ruta_config, 'w') as configfile:
                    config.write(configfile)

                hilos = []

                for i, (usuario, ip) in enumerate(dispositivos, start=1):
                    hilo = threading.Thread(target=codigo_hilo, args=(i, usuario, ip, ruta_excepcion, event,))
                    hilos.append(hilo)

                hilo = threading.Thread(target=codigo_local, args=(ruta_excepcion, event, config,))
                hilos.append(hilo)

                for hilo in hilos:
                    hilo.start()

                for hilo in hilos:
                    hilo.join()

                if event.is_set():
                    exit()
            
                last_round = get_last_round(config)
                step_rounds = rounds - (last_round - prev_rounds)

                if os.path.exists(ruta_excepcion):
                    os.remove(ruta_excepcion)

        # Asumiendo que textfile es una variable que contiene la ruta al archivo de texto
        textfile = "/home/usuario/Escritorio/Prueba.txt"

        # Abre el archivo de texto
        with open(textfile, 'r') as fp:  # Cambiado a 'r' ya que MIMEText espera una cadena de texto en modo de lectura normal, no binario
            msg = MIMEText(fp.read())

        # Información del remitente y destinatario
        me = 'victor26hid@gmail.com'  # Tu dirección de correo
        you = 'victor.hidalgo@uclm.es'  # Dirección del destinatario
        password = os.getenv('MY_APP_PASSWORD')  # La contraseña de tu cuenta o la contraseña de aplicación

        # Configurando los headers del email
        msg['Subject'] = f'Se ha terminado la ejecución {strategy} de FL.'
        msg['From'] = me
        msg['To'] = you

        # Enviar el mensaje usando el servidor SMTP de Gmail
        # Utiliza el servidor y puerto de tu proveedor de correo si no es Gmail
        s = smtplib.SMTP_SSL('smtp.gmail.com', 465)  # Cambiado para usar SSL
        s.login(me, password)  # Login con credenciales
        s.sendmail(me, [you], msg.as_string())  # Envío del email
        s.quit()


if __name__ == "__main__":
    main()