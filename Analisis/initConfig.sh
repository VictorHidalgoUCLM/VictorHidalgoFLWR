#!/bin/bash

# Definir las matrices de dispositivos remotos
REMOTE_HOSTS=("172.24.100.98" "172.24.100.10" "172.24.100.11")
REMOTE_USERS=("raspberrypi1" "raspberry4" "raspberry6")
REMOTE_DIRS=("/home/raspberrypi1" "/home/raspberry4" "/home/raspberry6")

# Directorio local a transferir
LOCAL_DIR="/home/usuario/Escritorio/Dockerizacion/data"

MAX_CLIENT_ID=$((${#REMOTE_HOSTS[@]} - 1))
NUM_CLIENTS=${#REMOTE_HOSTS[@]}
SCRIPT_PYTHON="/home/usuario/Escritorio/Dockerizacion/generarDatos.py"
MISMO_CLIENT_ID=$1
CLIENT_ID_UNICO=0

# Bucle for para ejecutar el script para cada dispositivo remoto
for ((i=0; i<=MAX_CLIENT_ID; i++)); do
    if [ "$MISMO_CLIENT_ID" = true ]; then
        CLIENT_ID=$CLIENT_ID_UNICO
    else
        CLIENT_ID=$i
    fi

    echo "Valor de client_id: $CLIENT_ID"

    python "$SCRIPT_PYTHON" --num_clients "$NUM_CLIENTS" --client_id "$CLIENT_ID"

    REMOTE_HOST="${REMOTE_HOSTS[$i]}"
    REMOTE_USER="${REMOTE_USERS[$i]}"
    REMOTE_DIR="${REMOTE_DIRS[$i]}"
    
    # Copiar el directorio utilizando scp
    scp -r "$LOCAL_DIR" "$REMOTE_USER@$REMOTE_HOST:$REMOTE_DIR"

    # Verificar el estado de salida de scp
    if [ $? -eq 0 ]; then
        echo "¡Directorio transferido con éxito a $REMOTE_HOST!"
    else
        echo "Error al transferir el directorio a $REMOTE_HOST."
    fi
done
