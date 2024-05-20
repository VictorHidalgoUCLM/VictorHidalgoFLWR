import numpy as np
import pandas as pd
import argparse
import os

from sklearn.model_selection import train_test_split
import tensorflow as tf

TEST_SIZE = 0.1428

parser = argparse.ArgumentParser()
parser.add_argument("--num_clients", type=int, help="Total clients")
parser.add_argument("--client_id", type=int, help="Client ID")
parser.add_argument("--data_type", type=int, help="Kind of distribution")

args = parser.parse_args()

client_id = args.client_id
num_clients = args.num_clients
data_type = args.data_type

if not os.path.exists("data"):
    os.makedirs("data")

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

if data_type == 0:
    x_part = np.array_split(np.concatenate([x_train, x_test]), num_clients)[client_id]
    y_part = np.array_split(np.concatenate([y_train, y_test]), num_clients)[client_id]

else:
    # Convertir a DataFrame
    x_part = np.concatenate([x_train, x_test])
    y_part = np.concatenate([y_train, y_test])

    x_part_df = pd.DataFrame(x_part.reshape(len(x_part), -1))
    y_part_df = pd.DataFrame(y_part, columns=["label"])

    indices = y_part_df["label"].sort_values().unique()
    indices_split = np.array_split(indices, num_clients)
    client_indices = indices_split[client_id]

    y_part = y_part_df[y_part_df['label'].isin(client_indices)]
    x_part = x_part_df.loc[y_part.index]

    y_part = y_part.reset_index(drop=True).values.flatten()
    x_part = x_part.reset_index(drop=True).values.reshape(-1, 28, 28)

x_train, x_test, y_train, y_test = train_test_split(x_part, y_part, test_size=TEST_SIZE)
x_train = tf.expand_dims(x_train, axis=-1)
x_test = tf.expand_dims(x_test, axis=-1)

# Convertir y_train y y_test en DataFrames
y_train_df = pd.DataFrame(y_train, columns=["label"])
y_test_df = pd.DataFrame(y_test, columns=["label"])

x_train = tf.image.resize(x_train, size=(32,32))
x_test = tf.image.resize(x_test, size=(32,32))

np.save(f"data/trainx", x_train)
np.save(f"data/trainy", y_train)
np.save(f"data/testx", x_test)
np.save(f"data/testy", y_test)