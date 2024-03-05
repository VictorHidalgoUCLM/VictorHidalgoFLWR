# Archivo: config.py
import os

config_path = os.path.expanduser("~/Escritorio/monitorizacion/master/prometheus.yml")
prometheus_url = 'http://localhost:9090'
image = "victorhidalgo/client"
sleep_time = 5