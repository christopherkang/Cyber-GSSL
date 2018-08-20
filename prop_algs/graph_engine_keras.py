import keras as k
import pandas as pd

NUM_OF_LABELS = 10

# EDGE_MATRIX format: columns are connections, rows are individual nodes
# values are weights
EDGE_MATRIX = pd.read_pickle("../data/pandas_weight_array.pickle")

# LABEL_LIST format: columns are labels and LL/LU/UU status
# rows are individual notes
LABEL_LIST = pd.read_pickle("I DONT KNOW THE FILE PATH")
TRAIN_STEPS = 100


def init(layer_list):

    model = k.models.Sequential()

    # add the input layer
    model.add(k.layers.Dense(500, activation="relu", input_shape=(500,)))

    # add the desired number of layers
    for layer in layer_list:
        model.add(k.layers.Dense(layer, activation="relu"))

    model.add(k.layers.Dense(NUM_OF_LABELS, activation="softmax"))

    return model


