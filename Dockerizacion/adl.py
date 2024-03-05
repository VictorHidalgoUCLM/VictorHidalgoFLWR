import requests
# Configuración para eliminar series de datos en el servidor
url = 'http://localhost:9090/api/v1/admin/tsdb/delete_series'
params = {'match[]': '{hostname="raspberrypi1"}'}

# Envía una solicitud HTTP POST para eliminar series de datos en el servidor
response = requests.post(url, params=params)