import os

import keras
import numpy as np
import pandas as pd
from keras import backend as K

os.chdir(os.path.dirname(__file__))
NUM_OF_LABELS = 10
EPOCHS = 100

# CONSTANTS
ALPHA_1 = 0.5
ALPHA_2 = 0.5
ALPHA_3 = 0.5

# EDGE_MATRIX format: columns are connections, rows are individual nodes
# values are weights
EDGE_MATRIX = pd.read_pickle("../data/pandas_weight_array.pickle")

# LABEL_LIST format: columns are labels and LL/LU/UU status
# rows are individual notes
# LABEL_LIST = pd.read_pickle("I DONT KNOW THE FILE PATH")
TRAIN_STEPS = 100


def get_neighbors(node_num):
    """Returns number of neighbors this node has

    Arguments:
        node_num {int} -- Node index

    Returns:
        int -- Number of nodes specified node is connected to
    """
    return np.count_nonzero(EDGE_MATRIX.loc[node_num])


def l1_norm(avec, bvec):
    """Returns the L1 norm of two vectors (l1 regularization squared )

    Arguments:
        avec {array} -- input vector
        bvec {array} -- input vector

    Returns:
        float -- L1 regularization squared
    """
    return K.sum(K.abs(avec-bvec), axis=-1, keepdims=True)


def l2_norm(avec, bvec):
    """Returns the L2 norm of two vectors (l2 regularization squared)

    Arguments:
        avec {array} -- input vector
        bvec {array} -- input vector

    Returns:
        float -- L2 regularization squared
    """
    return K.sum(K.square(avec-bvec), axis=-1, keepdims=True)


def c_x(index, actual):
    return (
        (1/get_neighbors(index)) *
        K.categorical_crossentropy(g_theta(EDGE_MATRIX.loc[index]), actual)
    )
    # NEED TO FIGURE OUT HOW TO MAKE g_theta = y_pred


def complex_loss(u, v):
    loss = 0
    if [LL]:
        loss += (ALPHA_1 * EDGE_MATRIX[u, v] * K.sum(l2_norm() +
                 c_x(u, FILLMEIN) + c_x(v, FILLMEIN)))
    elif [LU]:
        loss += (ALPHA_2 * EDGE_MATRIX[u, v] * K.sum(l2_norm() +
                 c_x(u, FILLMEIN)))
    elif [UU]:
        loss += ALPHA_3 * EDGE_MATRIX[u, v] * K.sum(l2_norm())
    else:
        raise Exception("FATAL_ERROR")


def init(layer_list):
    """intiates the model instance with specified inputs

    Arguments:
        layer_list {list} -- list of number of neurons in hidden layers

    Returns:
        Keras Sequential Model -- model, already compiled, and ready for use
    """

    model = keras.models.Sequential()

    # add the input layer
    model.add(keras.layers.Dense(500, activation="relu", input_shape=(500,)))

    # add the desired number of layers
    for layer in layer_list:
        model.add(keras.layers.Dense(layer, activation="relu"))

    # add final layer
    model.add(keras.layers.Dense(NUM_OF_LABELS, activation="softmax"))

    # compile - the loss/metrics functions will need to be changed
    model.compile(loss='MSE', optimizer='SGD', metrics=['accuracy'])
    return model


def one_hot(labels):
    """returns an array of one hot labels

    Arguments:
        labels {np array} -- input labels (categorical)

    Returns:
        np matrix -- one hot labels
    """

    one_hot_labels = keras.utils.to_categorical(labels,
                                                num_classes=NUM_OF_LABELS)
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
"""
for counter in range(EPOCHS):
    train(NN,  , )
"""
