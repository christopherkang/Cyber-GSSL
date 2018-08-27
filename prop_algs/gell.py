"""GRAPH ENGINE: LOW LEVEL (GELL)
"""

import pandas as pd
import tensorflow as tf

import readfile

# THIS IS NECESSARY FOR WINDOWS SYSTEMS
os.chdir(os.path.dirname(__file__))

# EDGE_MATRIX format: columns are connections, rows are individual nodes
# values are weights
EDGE_MATRIX = pd.read_pickle("../data/pandas_weight_array.pickle")

# LABEL_LIST format: columns are labels and LL/LU/UU status
# rows are individual notes
LABEL_LIST = pd.read_pickle("I DONT KNOW THE FILE PATH")

# THIS IS A NESTED LIST THAT DESCRIBES THE CONNECTIONS EACH NODE HAS
# BE MINDFUL THAT THE 0TH INDEX IS ASSOCIATED WITH AN IMAGINARY "0TH"
# NODE, AND THEREFORE HAS NO CONNECTIONS
NODE_CONNECTIONS = readfile.access_file("../data/node_connections.txt")

# THIS IS A NESTED LIST THAT DESCRIBES THE TYPE OF EACH EDGE
TOTAL_LLUU_LIST = readfile.access_file("../data/total_edge_type.txt")

# THESE VARIABLES ARE NOW OBSOLETE AND ARE KEPT FOR DEBUGGING
LIST_OF_LABELED_EDGES = TOTAL_LLUU_LIST[0]
LIST_OF_MIXED_EDGES = TOTAL_LLUU_LIST[1]
LIST_OF_ALONE_EDGES = TOTAL_LLUU_LIST[2]

# NUMBER OF LABELS - THIS SHOULD BE DYNAMICALLY ASSIGNED IN THE FUTURE
NUM_OF_LABELS = 10

# NUMBER OF STEPS TO TRAIN
TRAIN_STEPS = 100

# ALPHAs - THESE ARE USED AS CONSTANTS IN MULTIPLICATION
ALPHA_1 = tf.constant(0.5, dtype=tf.float32, name="ALPHA_1")
ALPHA_2 = tf.constant(0.5, dtype=tf.float32, name="ALPHA_2")
ALPHA_3 = tf.constant(0.5, dtype=tf.float32, name="ALPHA_3")


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


def h_theta(index, total_matrix):
    return total_matrix[tf.where(tf.equal(total_matrix[:, 0], index)), -1]


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


def custom_loss(labels, predicted, label_type_list):

    temp_sum = tf.convert_to_tensor(0)
    # iterate through each type of edge
    for u_pair, v_pair in label_type_list[0]:
        # perform ALPHA 1 loss
        # temp_sum = tf.add(temp_sum, tf.reduce_sum(ALPHA_1*))
        temp_sum += tf.reduce_sum(
            ALPHA_1 * EDGE_MATRIX[u_pair, v_pair] *
            tf.norm(h_theta(u_pair, predicted)-h_theta(v_pair, predicted)) +
            c_x(u_pair, labels[u_pair]) + c_x(v_pair, labels[v_pair]))

    for u_mixed, v_mixed in label_type_list[1]:
        # temp_sum = tf.add(temp_sum, tf.reduce_sum(ALPHA_2*))
        temp_sum += tf.reduce_sum(
            ALPHA_2 * EDGE_MATRIX[u_mixed, v_mixed] *
            tf.norm(h_theta(u_mixed, predicted) -
                    h_theta(v_mixed, predicted)) + c_x(
                        u_mixed, labels[u_mixed]))

    for u_alone, v_alone in label_type_list[2]:
        # temp_sum = tf.add(temp_sum, tf.reduce_sum(ALPHA_3*))
        temp_sum += tf.reduce_sum(
            ALPHA_3 * EDGE_MATRIX[u_alone, v_alone] *
            tf.norm(h_theta(u_alone, predicted)-h_theta(v_alone, predicted)))

    return temp_sum


def make_feature_col(features, range):
    temp_feature_cols = set()
    for col in range(range[0], range[1]):
        temp_feature_cols += tf.feature_column.numeric_column(
            features.columns.values[col])
    return temp_feature_cols


def my_model_fn(dataset, hidden_nodes):

    net = tf.feature_column.input_layer(
        dataset[:, 1:-1], make_feature_col(
            EDGE_MATRIX, [0, EDGE_MATRIX.shape[1]]))
    for units in hidden_nodes:
        # then, pass the output through the hidden layers
        net = tf.layers.dense(net, units=units, activation=tf.nn.relu)

    # the tf.nn.softmax can't be used as an activation function, so
    # it is applied afterwords
    logits = tf.layers.dense(
        net, params['n_classes'], activation=tf.nn.softmax)

    # everything except labels (pred at end)
    comb_mat = tf.concat([logits, dataset[:, :-1]], 0)

    # give two datasets - one has the labels, the other has the reps
    loss = custom_loss(dataset[:, -1], comb_mat, TOTAL_LLUU_LIST)
    optimizer = tf.train.GradientDescentOperator(0.01)
    train = optimizer.minimize(loss)

    init = tf.global_variables_initializer()

    sess = tf.Session()
    sess.run(init)

    for counter in range(100):
        _, loss_value = sess.run((train, loss))
        print(loss_value)

# THE DATASET IS COMPRISED OF INDEX VALUES TO IDENTIFY THE NODES,
# THE EDGE WEIGHTS, AND THE LABELS

slices = tf.data.Dataset.from_tensor_slices(
    (EDGE_MATRIX.index.values, EDGE_MATRIX.values, LABEL_LIST.values))
slices = slices.shuffle()
slices = slices.batch(30).repeat(count=None)
next_item = slices.make_one_shot_iterator().get_next()
