import flwr as fl
import tensorflow as tf

import numpy as np
import math
import random

# Definición del modelo MobileNet para clasificación en CIFAR-10
model = tf.keras.applications.MobileNet((32, 32, 1), classes=10, weights=None)
model.compile("adam", "sparse_categorical_crossentropy", metrics=["accuracy"])


class CifarClient(fl.client.NumPyClient):
    """Cliente personalizado para el entrenamiento federado con CIFAR-10"""

    def get_parameters(self, config):
        """Obtiene los pesos del modelo."""
        return model.get_weights()

    def fit(self, parameters, config):
        """Realiza el entrenamiento local en el cliente."""

        # Configura el modelo con los pesos proporcionados
        model.set_weights(parameters)

        # Carga los datos de entrenamiento
        x_train = np.load(f"/data/trainx.npy")
        y_train = np.load(f"/data/trainy.npy")

        # Obtiene los parámetros de configuración
        subset_size = config["subset_size"]
        batch_size = config["batch_size"]
        epochs = config["epochs"]
        evaluate_on_fit = config["evaluate_on_fit"]
        server_round = config["server_round"]

        # Selecciona un subconjunto aleatorio de los datos de entrenamiento
        """inicio = (server_round - 1) * subset_size
        fin = server_round * subset_size

        inicio = inicio % len(x_train)
        fin = fin % len(x_train)"""

        inicio = 0
        fin = subset_size

        # Seleccionar la ventana específica de los datos de entrenamiento
        if inicio < fin:
            x_train = x_train[inicio:fin]
            y_train = y_train[inicio:fin]
        else:
            x_train = x_train[inicio:] + x_train[:fin]
            y_train = y_train[inicio:] + y_train[:fin]

        # Entrenamiento del modelo local
        history = model.fit(
            x_train,
            y_train,
            epochs=epochs,
            batch_size=batch_size,
            steps_per_epoch=math.ceil(len(x_train) / batch_size)
        )

        # Obtiene la última precisión registrada durante el entrenamiento
        accuracy = history.history['accuracy'][-1]
        loss_distributed = history.history['loss'][-1]

        if evaluate_on_fit:
            temp_param = model.get_weights()

            loss, _, _ = self.evaluate(temp_param, {})
        
        else:
            loss = 0.0

        # Devuelve los pesos actualizados, tamaño del subconjunto y métricas
        return model.get_weights(), len(x_train), {"accuracy": float(accuracy), "loss": float(loss), "loss_distributed": float(loss_distributed)}

    def evaluate(self, parameters, config):
        """Evalúa el modelo en datos de prueba."""

        # Configura el modelo con los pesos proporcionados
        model.set_weights(parameters)

        # Carga los datos de prueba
        x_test = np.load(f"/data/testx.npy")
        y_test = np.load(f"/data/testy.npy")

        # Evaluación del modelo en datos de prueba
        loss, accuracy = model.evaluate(x_test, y_test)

        # Devuelve la pérdida, tamaño del conjunto de prueba y métricas
        return loss, len(x_test), {"accuracy": float(accuracy)}


# Inicia el cliente federado
fl.client.start_numpy_client(server_address="172.24.100.129:8080",
                             client=CifarClient(),
                             )
