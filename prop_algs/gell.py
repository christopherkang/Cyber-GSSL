"""GRAPH ENGINE: LOW LEVEL (GELL)
"""

import tensorflow as tf
import pandas as pd

import readfile

os.chdir(os.path.dirname(__file__))

# EDGE_MATRIX format: columns are connections, rows are individual nodes
# values are weights
EDGE_MATRIX = pd.read_pickle("../data/pandas_weight_array.pickle")
# LABEL_LIST format: columns are labels and LL/LU/UU status
# rows are individual notes
LABEL_LIST = pd.read_pickle("I DONT KNOW THE FILE PATH")

NODE_CONNECTIONS = readfile.access_file("../data/node_connections.txt")
TOTAL_LLUU_LIST = readfile.access_file("../data/total_edge_type.txt")
LIST_OF_LABELED_EDGES = TOTAL_LLUU_LIST[0]
LIST_OF_MIXED_EDGES = TOTAL_LLUU_LIST[1]
LIST_OF_UNLABELED_EDGES = TOTAL_LLUU_LIST[2]

NUM_OF_LABELS = 10

TRAIN_STEPS = 100

# ALPHAs
ALPHA_1 = tf.constant(0.5, dtype=tf.float32, name="ALPHA_1")
ALPHA_2 = tf.constant(0.5, dtype=tf.float32, name="ALPHA_2")
ALPHA_3 = tf.constant(0.5, dtype=tf.float32, name="ALPHA_3")

# TF Variables



def g_theta(index):
    gtemp = None
    # for each node the node is connected to
    temp_label_hold = []
    sum_weights = {}
    for neighbors in NODE_CONNECTIONS[index]:
        if LABEL_LIST[neighbors] == -1:  # only count "real" labels
            pass
        else:
            label_val = EDGE_MATRIX[index][neighbors]
            try:
                sum_weights[LABEL_LIST[neighbors]] += label_val
            except:
                sum_weights[LABEL_LIST[neighbors]] = label_val
    try:
        max_value = max(sum_weights.values())
        max_value = list({key for key, value in sum_weights.items()
                          if value == max_value})
    except:
        return -1
    print("these are in top_labels ", max_value)
    # a set of the most popular labels
    return tf.convert_to_tensor(random.choice(max_value))


def h_theta(index):
    pass


def get_neighbors(index):
    """Returns number of neighbors this node has

    Arguments:
        index {int} -- Node index

    Returns:
        int -- Number of nodes specified node is connected to
    """

    return tf.convert_to_tensor(np.count_nonzero(EDGE_MATRIX.loc[index]))


def c_x(index, labels):

    return tf.convert_to_tensor(
              (1/get_neighbors(index)) *
              tf.reduce_sum(tf.reduce_mean(labels * tf.log(g_theta(index)))))


def custom_loss(u, v, labels, ):

    temp_sum = tf.convert_to_tensor(0)
    # iterate through each type of edge
    for u_pair, v_pair in LIST_OF_LABELED_EDGES:
        # perform ALPHA 1 loss
        # temp_sum = tf.add(temp_sum, tf.reduce_sum(ALPHA_1*))
        temp_sum += tf.reduce_sum(
            ALPHA_1 * EDGE_MATRIX[u_pair, v_pair] *
            tf.norm(h_theta(u_pair)-h_theta(v_pair)) +
            c_x(u, labels[u]) + c_x(v, labels[v]))

    for u_mixed, v_mixed in LIST_OF_MIXED_EDGES:
        # temp_sum = tf.add(temp_sum, tf.reduce_sum(ALPHA_2*))
        temp_sum += tf.reduce_sum(
            ALPHA_2 * EDGE_MATRIX[u_mixed, v_mixed] *
            tf.norm(h_theta(u_mixed)-h_theta(v_mixed)))

    for u_alone, v_alone in LIST_OF_ALONE_EDGES:
        # temp_sum = tf.add(temp_sum, tf.reduce_sum(ALPHA_3*))
        temp_sum += tf.reduce_sum(
            ALPHA_3 * EDGE_MATRIX[u_alone, v_alone] *
            tf.norm(h_theta(u_alone)-h_theta(v_alone)))

    return temp_sum


def make_feature_col(features, range):
    temp_feature_cols = set()
    for col in range(range[0], range[1]):
        temp_feature_cols += tf.feature_column.numeric_column(
            features.columns.values[col])
    return temp_feature_cols


def my_model_fn(dataset, hidden_nodes):

    net = tf.feature_column.input_layer(
        dataset, make_feature_col(EDGE_MATRIX, [0, EDGE_MATRIX.shape[1]]))
    for units in params['hidden_units']:
        # then, pass the output through the hidden layers
        net = tf.layers.dense(net, units=units, activation=tf.nn.relu)

    # the tf.nn.softmax can't be used as an activation function, so
    # it is applied afterwords
    logits = tf.layers.dense(
        net, params['n_classes'], activation=tf.nn.softmax)




# THE DATASET IS COMPRISED OF INDEX VALUES TO IDENTIFY THE NODES,
# THE EDGE WEIGHTS, AND THE LABELS

slices = tf.data.Dataset.from_tensor_slices(
    (EDGE_MATRIX.index.values, EDGE_MATRIX.values, LABEL_LIST.values))

slices = slices.batch(30).repeat(count=None)
next_item = slices.make_one_shot_iterator().get_next()
