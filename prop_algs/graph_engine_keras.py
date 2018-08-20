import os

import keras as k
import pandas as pd

os.chdir(os.path.dirname(__file__))
NUM_OF_LABELS = 10
EPOCHS = 100
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


def one_hot(labels):
    """returns an array of one hot labels

    Arguments:
        labels {np array} -- input labels (categorical)

    Returns:
        np matrix -- one hot labels
    """

    one_hot_labels = k.utils.to_categorical(labels, num_classes=NUM_OF_LABELS)
    return one_hot_labels


def train(neural_net, input_features, input_labels):
    """iteratively train a neural network

    Arguments:
        neural_net {model from K} -- model of neural network
        input_features {np array?} -- input features
        input_labels {matrix} -- matrix of features
    """

    neural_net.fit(input_features, input_labels,
                   epochs=1, batch_size=input_features.shape[0])


LABEL_LIST = one_hot(LABEL_LIST)

NN = init([500, 50])

for counter in range(EPOCHS):
    train(NN,  , )
