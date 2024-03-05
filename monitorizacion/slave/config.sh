#!/bin/bash

# Actualizar el índice de paquetes
sudo apt-get update

# Instalar Docker y Docker Compose
sudo apt-get install -y docker docker-compose

cmdline="/boot/cmdline.txt"
nuevas_opciones="cgroup_enable=cpu_set cgroup_enable=memory cgroup_memory=1 swapaccount=1"

sudo sed -i "s/\(.*\)/\1 $nuevas_opciones/" "$archivo_configuracion"

# Crear grupo docker y añadir al usuario actual
sudo groupadd docker
sudo gpasswd -a $USER docker
newgrp docker

# Definir el archivo de configuración
archivo_configuracion="slave/.env"

# Guardar las respuestas en el archivo de configuración
echo "pais=ES" > "$archivo_configuracion"
echo "entorno=RP4" >> "$archivo_configuracion"
echo "cliente=victor" >> "$archivo_configuracion"
echo "hostname=raspberrypi4" >> "$archivo_configuracion"

