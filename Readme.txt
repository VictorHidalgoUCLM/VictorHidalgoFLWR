Este repositorio público almacena el código desarrollado e implementado para el uso de Federated Learning 
a través de Flwr Framework para el Trabajo Fin de Máster (TFM) "Despliegue de una herramienta de aprendizaje 
federado sobre una infraestructura distribuida de bajo coste".

Analisis -> Código destinado a crear gráficas sobre las métricas y resultados de las ejecuciones para 
            analizar los resultados.

Automatización -> Scripts modificados sobre los originales de Flwr para modificar el comportamiento originales, 
            así como un script de ejecución automática "run" que realiza automáticamente varias ejecuciones de 
            varias estrategias.

devices_config -> Código destinado a repartir los datos de forma automática entre los clientes, así como configurar
            el servidor para el aprendizaje federado ante nuevas configuraciones.

Dockerizacion -> Definición de contenedores docker que contienen el código del cliente, de forma automática actualiza
            el código modificado y crea el contenedor, subiéndolo a DockerHub para ser utilizado más tarde.

results -> Directorio que almacena resultados "raw" de la ejecución de diversas estrategias. Cuando se ejecutan los 
            scripts de análisis, las gráficas quedan almacenadas también aquí.