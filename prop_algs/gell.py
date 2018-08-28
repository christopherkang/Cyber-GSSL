"""GRAPH ENGINE: LOW LEVEL (GELL)
"""

import numpy as np
import pandas as pd
import tensorflow as tf

import os
import readfile

# THIS IS NECESSARY FOR WINDOWS SYSTEMS
os.chdir(os.path.dirname(__file__))

# EDGE_MATRIX format: columns are connections, rows are individual nodes
# values are weights
EDGE_MATRIX = pd.read_pickle("../data/pandas_weight_array.pickle")

# LABEL_LIST format: columns are labels and LL/LU/UU status
# rows are individual notes
# LABEL_LIST = pd.read_pickle("I DONT KNOW THE FILE PATH")
LABEL_LIST = pd.DataFrame(
    np.zeros(500)-1, index=range(1, 501), columns=["LABELS"])

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
        if LABEL_LIST.loc[neighbors] == -1:  # only count "real" labels
            pass
        else:
            label_val = EDGE_MATRIX[index][neighbors]
            try:
                sum_weights[LABEL_LIST.loc[neighbors]] += label_val
            except:
                sum_weights[LABEL_LIST.loc[neighbors]] = label_val
    try:
        max_value = max(sum_weights.values())
        max_value = list({key for key, value in sum_weights.items()
                          if value == max_value})
    except:
        return -1
    print("these are in top_labels ", max_value)
    # a set of the most popular labels
    return tf.convert_to_tensor(random.choice(max_value))


def g_theta_total(index):
    """Produces a NUM_OF_LABELS by 1 vector describing the probs of each label

    Arguments:
        index {int} -- integer index of the specified node

    Returns:
        tf tensor -- tensor with the values of the probabilities
    """

    average_prob_value = np.zeros((NUM_OF_LABELS, 1))
    for neighbors in NODE_CONNECTIONS[index]:
        if LABEL_LIST.loc[neighbors] == -1:
            pass
        else:
            average_prob_value[LABEL_LIST.loc[neighbors]] += EDGE_MATRIX[
                index, neighbors]
    if sum(average_prob_value) == 0:
        return tf.convert_to_tensor(
            np.zeros((NUM_OF_LABELS, 1)) + 1/NUM_OF_LABELS)
    return tf.convert_to_tensor(average_prob_value/sum(average_prob_value))


def find_value(index, vector):
    co_ords = tf.where(tf.equal(vector, index), name="find_value_fn")[0]
    return vector[co_ords[0]]


def h_theta(index, total_matrix):
    """Returns the H_theta value associated with the loss function
    Essentially the output of the neural network

    Arguments:
        index {int} -- the node's index
        total_matrix {tf tensor} -- tensor with all of the predictions

    Returns:
        tf tensor -- tensor output of the neural network for a specific index
    """

    return tf.convert_to_tensor(
        total_matrix[total_matrix, index, 1:(NUM_OF_LABELS+1)])


def d_term_h(index_u, index_v, nn_output):
    return tf.norm(nn_output[index_u] - nn_output[index_v])


def get_neighbors(index):
    """Returns number of neighbors this node has

    Arguments:
        index {int} -- Node index

    Returns:
        int -- Number of nodes specified node is connected to
    """

    return tf.convert_to_tensor(np.count_nonzero(EDGE_MATRIX.loc[index]))


def c_x(index, labels):
    """This function finds the cross entropy described in the loss fn

    Arguments:
        index {tf tensor} -- should be a vector with the indices of relevant
            nodes
        labels {tf tensor} -- should be a vector with the actual labels of
            the relevant nodes. It should NOT be a one-hot vector.

    Returns:
        tf tensor -- returns a tensor of all the answers
    """

    # THIS FUNCTION IS RELATIVELY COMPLEX: LET'S BREAK IT DOWN
    # FIRST, WE ARE CONVERTING THE VALUE TO A TENSOR
    # THEN, WE USE MAP_FN TO APPLY GET_NEIGHBORS TO EACH ELEMENT IN INDEX
    # NEXT, WE NEED TO SUM OVER THE PRODUCT AND LABELS
    # USING ONE_HOT, WE CAN CREATE A ONE HOT PROB VECTOR WITH 1 AS TRUE
    # WE MULTIPLY!
    return tf.convert_to_tensor((1/get_neighbors(index))) * tf.reduce_sum(
                  tf.one_hot(labels, depth=NUM_OF_LABELS) *
                  tf.log(g_theta_total(index)))


def custom_loss(labels, predicted, reference_vector, label_type_list):
    """The custom loss function for the NN

    Arguments:
        labels {tf tensor} -- list of labels
        predicted {tf tensor} -- list of predicted labels
        label_type_list {pandas DF} -- DF with all of the edge types

    Returns:
        tf tensor -- scalar of the final loss value
    """

    # RELATIVE INDEX VARS ARE THE INDICES OF THE NODE IN LABELS/PREDICTED VECS
    temp_sum = tf.convert_to_tensor(0)

    # iterate through each type of edge
    for u_pair, v_pair in label_type_list[0]:
        relative_index_u = find_value(u_pair, reference_vector)
        relative_index_v = find_value(v_pair, reference_vector)
        temp_sum += ALPHA_1 * tf.reduce_sum(
            EDGE_MATRIX.loc[u_pair, v_pair] *
            d_term_h(relative_index_u, relative_index_v, predicted) +
            c_x(u_pair, labels[u_pair]) + c_x(v_pair, labels[v_pair]))

    for u_mixed, v_mixed in label_type_list[1]:
        # temp_sum = tf.add(temp_sum, tf.reduce_sum(ALPHA_2*))
        relative_index_u = find_value(u_mixed, reference_vector)
        relative_index_v = find_value(v_mixed, reference_vector)
        temp_sum += ALPHA_2 * tf.reduce_sum(
            EDGE_MATRIX.loc[u_mixed, v_mixed] *
            d_term_h(relative_index_u, relative_index_v, predicted) +
            c_x(u_mixed, labels[u_mixed]))

    for u_alone, v_alone in label_type_list[2]:
        # temp_sum = tf.add(temp_sum, tf.reduce_sum(ALPHA_3*))
        relative_index_u = find_value(u_alone, reference_vector)
        relative_index_v = find_value(v_alone, reference_vector)
        temp_sum += ALPHA_3 * tf.reduce_sum(
            EDGE_MATRIX.loc[u_alone, v_alone] *
            d_term_h(relative_index_u, relative_index_v, predicted))

    return temp_sum


def make_feature_col(features, inp_range):
    temp_feature_cols = []
    for col in range(inp_range[0], inp_range[1]):
        temp_feature_cols.append(tf.feature_column.numeric_column(
            str(features.columns.values[col])))
    return temp_feature_cols


def make_dict_feature_col(dict_of_features):
    return [tf.feature_column.numeric_column(str(col))
            for col in dict_of_features]


def my_model_fn(dataset, hidden_nodes):
    """NN Model function

    Arguments:
        dataset {pd DF} -- dataset with indices
        hidden_nodes {list} -- list of hidden_nodes
    """

    # THIS CREATES THE INPUT LAYER AND IMPORTS DATA
    net = tf.feature_column.input_layer(
        dataset[0], make_dict_feature_col(dataset[0]))

    # BUILDS HIDDEN LAYERS
    for units in hidden_nodes:
        net = tf.layers.dense(net, units=units, activation=tf.nn.relu)

    # BUILDS THE FINAL LAYERS
    logits = tf.layers.dense(
        net, NUM_OF_LABELS, activation=tf.nn.softmax)

    # everything except labels (pred at end)
    # this makes a larger matrix with the indices, logit outputs, and inputs
    # comb_mat = tf.concat([dataset[1], dataset[0], logits], 0)

    # give two datasets - one has the labels, the other has the reps
    loss = custom_loss(dataset[2], logits, dataset[1], TOTAL_LLUU_LIST)
    optimizer = tf.train.GradientDescentOperator(0.01)
    train = optimizer.minimize(loss)

    init = tf.global_variables_initializer()

    with tf.Session() as sess:
        writer = tf.summary.FileWriter("./tmp/log/...", sess.graph)
        sess.run(init)
        for counter in range(100):
            _, loss_value = sess.run((train, loss))
            print(loss_value)
        writer.close()

# THE DATASET IS COMPRISED OF INDEX VALUES TO IDENTIFY THE NODES,
# THE EDGE WEIGHTS, AND THE LABELS

# TEMP CODE FOR DEBUGGING
# EDGE_MATRIX.insert(0, column="index", value=EDGE_MATRIX.index.values)
# EDGE_MATRIX['label'] = LABEL_LIST.values

# ------- ! BEGIN DATA IMPORT PIPELINE ! ------- #
# slices = tf.data.Dataset.from_tensor_slices(EDGE_MATRIX)

# THE USE OF TUPLES IS PREFERABLE, AS IN THE END EVERYTHING IS A DICT

zipped_features = {str(key): np.array(value)
                   for key, value in dict(EDGE_MATRIX).items()}

slices = tf.data.Dataset.from_tensor_slices(
    (zipped_features, np.arange(start=1, stop=501), LABEL_LIST.values))

# slices = slices.shuffle(100)
# slices = slices.batch(len(EDGE_MATRIX.index.values)).repeat(count=None)

slices = slices.batch(1).repeat(count=None)

# THIS WILL OUTPUT, ROW BY ROW, THE NODES
next_item = slices.make_one_shot_iterator().get_next()

# ------- ! END DATA IMPORT PIPELINE ! ------- #

"""
# DEBUGGING CODE
# print(np.transpose(EDGE_MATRIX.iloc[:1, :].values).tolist())
with tf.Session() as sess:
    print(sess.run(next_item[0]))
    # print(sess.run(next_item))
"""



my_model_fn(next_item, [500, 500, 20])
