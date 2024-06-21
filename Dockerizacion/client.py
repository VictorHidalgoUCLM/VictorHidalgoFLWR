"""
This code runs one client in the Federated Learning (FL) process,
this exact same code is on each of the clients. The CifarClient class
defines the functions that the server will be able to call.

There are three functions that allow remote fit on each client (fit function),
remote evaluation (evaluate function) and to send model weights (get_parameters)
to the server so it can aggregate all of the parameters.

At the end of the code, the client service starts and tries to connect to the
server through provided IP address.
"""

# Necessary modules
import flwr as fl
import tensorflow as tf
from tensorflow.python.keras.regularizers import l1_l2

import tensorflow.keras.backend as K

import numpy as np
import math

import gc

def recall_f(y_true, y_pred):
    # Obtener las clases predichas
    y_pred_classes = tf.argmax(y_pred, axis=-1)

    # Convertir y_true a un formato que se pueda comparar
    y_true = tf.cast(y_true, y_pred_classes.dtype)

    # Calcular verdaderos positivos y posibles positivos por clase
    recalls = []

    for class_label in range(10):  # 10 clases en total
        true_positives = K.sum(K.cast((y_true == class_label) & (y_pred_classes == class_label), dtype=tf.float32))
        true_classes = K.sum(K.cast(y_true == class_label, dtype=tf.float32))
        recalls.append(true_positives / (true_classes + K.epsilon()))
        
    # Calcular el recall ponderado
    recall = tf.reduce_sum(recalls) / 10

    return recall


def precision_f(y_true, y_pred):
    # Convertir probabilidades en predicciones de clases
    y_pred_classes = tf.argmax(y_pred, axis=-1)
    
    # Convertir y_true a un formato que se pueda comparar
    y_true = tf.cast(y_true, y_pred_classes.dtype)
    
    # Calcular verdaderos positivos y falsos positivos por clase
    precisions = []

    for class_label in range(10):  # 10 clases en total
        true_positives = K.sum(K.cast((y_true == class_label) & (y_pred_classes == class_label), dtype=tf.float32))
        precition_classes = K.sum(K.cast(y_pred_classes == class_label, dtype=tf.float32))
        precisions.append(true_positives / (precition_classes + K.epsilon()))

    precision = tf.reduce_sum(precisions) / 10

    return precision


class CifarClient(fl.client.NumPyClient):
    """Client class modified to get extra config hiperparameters"""


    def __init__(self):
        """Initializes the client with an empty model"""
        self.model = tf.keras.applications.MobileNet((32, 32, 1), classes=10, weights=None)

    def get_parameters(self, config):
        """Returns local model weights"""
        return self.model.get_weights()

    def fit(self, parameters, config):
        """Fits the model with its own local data.
        Gets the initial parameters of the model and the
        initial config, such as subset_size, epochs..."""

        # Obtains initial config for this client
        subset_size = config["subset_size"]                 # Subset size of data for training
        batch_size = config["batch_size"]                   # Batch size for training
        epochs = config["epochs"]                           # Number of epochs for training
        evaluate_on_fit = config["evaluate_on_fit"]         # To evaluate right after training
        server_round = config["server_round"]               # Current server_round
        proximal_mu = config.get("proximal_mu", 0.0)       # Only for FedProx, else defaults to 0

        if server_round == 1:
            if proximal_mu != 0.0:
                # Adds regularizer l1_l2 to the model
                regularizer = l1_l2(l1=0, l2=proximal_mu)

                # For each layer, regularizer is updated
                for layer in self.model.layers:
                    if hasattr(layer, 'kernel_regularizer'):
                        layer.kernel_regularizer = regularizer

            self.model.compile("adam", "sparse_categorical_crossentropy", metrics=["accuracy"])

        # Charges initial parameters provided by server
        self.model.set_weights(parameters)

        # Charges local-train data
        x_train = np.load(f"/data/trainx.npy")
        y_train = np.load(f"/data/trainy.npy")

        # Selects subset given the current training round
        start = (server_round - 1) * subset_size
        end = server_round * subset_size

        start = start % len(x_train)
        end = end % len(x_train)

        # Selects local data correctly in case 'start' > 'fin'
        if start < end:
            x_train_subset = x_train[start:end]
            y_train_subset = y_train[start:end]
        else:
            x_train_subset = np.concatenate((x_train[start:], x_train[:end]))
            y_train_subset = np.concatenate((y_train[start:], y_train[:end]))

        del x_train, y_train
        gc.collect()

        # Local model training, using read config
        self.model.fit(
            x_train_subset,
            y_train_subset,
            epochs=epochs,
            batch_size=batch_size,
            steps_per_epoch=math.ceil(len(x_train_subset) / batch_size / epochs)
        )

        loss_distributed, accuracy = self.model.evaluate(x_train_subset, y_train_subset, verbose=0)
        y_pred = self.model.predict(x_train_subset)

        #   Calculate metrics
        precision_value = precision_f(y_train_subset, y_pred)
        recall_value = recall_f(y_train_subset, y_pred)
        f1 = 2 * (precision_value * recall_value) / (precision_value + recall_value + K.epsilon())

        len_xtrain = len(x_train_subset)
        loss = 0.0

        del x_train_subset, y_train_subset, y_pred
        gc.collect()

        # Returns model weights, number of data used to train
        # and metrics for the server
        return self.model.get_weights(), len_xtrain, {"accuracy": float(accuracy), "loss": float(loss), "loss_distributed": float(loss_distributed), "recall": float(recall_value), "precision": float(precision_value), "f1_score": float(f1)}

    def evaluate(self, parameters, config):
        """Evaluates a given model with data test"""
        self.model.set_weights(parameters)

        # Charges local-data test
        x_test = np.load(f"/data/testx.npy")
        y_test = np.load(f"/data/testy.npy")
            
        loss, accuracy = self.model.evaluate(x_test, y_test, verbose=0)
        y_pred = self.model.predict(x_test)
    
        #   Calculate metrics
        precision_value = precision_f(y_test, y_pred)
        recall_value = recall_f(y_test, y_pred)
        f1 = 2 * (precision_value * recall_value) / (precision_value + recall_value + K.epsilon())
    
        len_xtest = len(x_test)

        del x_test, y_test, y_pred
        gc.collect()

        # Returns loss
        return loss, len_xtest, {"accuracy": float(accuracy), "recall": float(recall_value), "precision": float(precision_value), "f1_score": float(f1)}


# Starts federated client
fl.client.start_client(server_address="172.24.100.136:8080",  # Server IP
                             client=CifarClient().to_client(),  # Client code is new instace of CifarClient class
                             )
