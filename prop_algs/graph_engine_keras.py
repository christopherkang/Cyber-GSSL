import keras as k
import pandas as pd
import os

os.chdir(os.path.dirname(__file__))
NUM_OF_LABELS = 10

# EDGE_MATRIX format: columns are connections, rows are individual nodes
# values are weights
EDGE_MATRIX = pd.read_pickle("../data/pandas_weight_array.pickle")

# LABEL_LIST format: columns are labels and LL/LU/UU status
# rows are individual notes
LABEL_LIST = pd.read_pickle("I DONT KNOW THE FILE PATH")
TRAIN_STEPS = 100


def init(layer_list):
    """intiates the model instance with specified inputs

    Arguments:
        layer_list {list} -- list of number of neurons in hidden layers

    Returns:
        Keras Sequential Model -- model, already compiled, and ready for use
    """

    model = k.models.Sequential()

    # add the input layer
    model.add(k.layers.Dense(500, activation="relu", input_shape=(500,)))

    # add the desired number of layers
    for layer in layer_list:
        model.add(k.layers.Dense(layer, activation="relu"))

    model.add(k.layers.Dense(NUM_OF_LABELS, activation="softmax"))

    model.compile(loss='MSE', optimizer='SGD', metrics=['accuracy'])
    return model


def train(neural_net, input_features, input_labels):

    neural_net.fit(input_features, input_labels, epochs=1,
                   batch_size=input_features.shape[0])

NN = init([500, 50])
