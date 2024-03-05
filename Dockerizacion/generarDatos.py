import numpy as np
import argparse
import os
import cv2
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras import layers

TEST_SIZE = 0.1428

parser = argparse.ArgumentParser()
parser.add_argument("--num_clients", type=int, help="Cantidad de clientes")
parser.add_argument("--client_id", type=int, help="Número de cliente")

args = parser.parse_args()

client_id = args.client_id
num_clients = args.num_clients

if not os.path.exists("data"):
    os.makedirs("data")

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

x_part = np.array_split(np.concatenate([x_train, x_test]), num_clients)[client_id]
y_part = np.array_split(np.concatenate([y_train, y_test]), num_clients)[client_id]

x_train, x_test, y_train, y_test = train_test_split(x_part, y_part, test_size=TEST_SIZE)
x_train = tf.expand_dims(x_train, axis=-1)
x_test = tf.expand_dims(x_test, axis=-1)

x_train = tf.image.resize(x_train, size=(32,32))
x_test = tf.image.resize(x_test, size=(32,32))

np.save(f"data/trainx", x_train)
np.save(f"data/trainy", y_train)
np.save(f"data/testx", x_test)
np.save(f"data/testy", y_test)
