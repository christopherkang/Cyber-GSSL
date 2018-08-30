"""GRAPH ENGINE: LOW LEVEL (GELL)
"""

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.python import debug as tf_debug

import os
import readfile

# THIS IS NECESSARY FOR WINDOWS SYSTEMS
os.chdir(os.path.dirname(__file__))

# EDGE_MATRIX format: columns are connections, rows are individual nodes
# values are weights
EDGE_MATRIX = pd.read_pickle("../data/pandas_weight_array.pickle")

# LABEL_LIST format: columns are labels and LL/LU/UU status
# rows are individual notes
# imported_labels = pd.read_pickle("I DONT KNOW THE FILE PATH")
# imported labels are the CVE valued ones (real_df)
# list_of_CWEs = set(real_df.index)
# refined_list = pd.DataFrame(
#   np.arange(0, len(list_of_CWEs)),
#   index=sorted(list(list_of_CWEs)), columns=["NEW_ID"])
# CWE_to_index = refined_list.to_dict()["NEW_ID"]
# LABEL_LIST = pd.read_pickle(USE DATASET WITH CWE VALUES)
# LABEL_LIST.replace(CWE_to_index)
LABEL_LIST = pd.DataFrame(
    np.zeros(500)-1, index=range(1, 501), columns=["LABELS"])
LABEL_LIST.loc[4, "LABELS"] = 1
LABEL_LIST = LABEL_LIST.astype(int)

# THIS IS A NESTED LIST THAT DESCRIBES THE CONNECTIONS EACH NODE HAS
# BE MINDFUL THAT THE 0TH INDEX IS ASSOCIATED WITH AN IMAGINARY "0TH"
# NODE, AND THEREFORE HAS NO CONNECTIONS
NODE_CONNECTIONS = readfile.access_file("../data/node_connections.txt")

# THIS IS A NESTED LIST THAT DESCRIBES THE TYPE OF EACH EDGE
TOTAL_LLUU_LIST = readfile.access_file("../data/total_edge_type.txt")

# NUMBER OF LABELS - THIS SHOULD BE DYNAMICALLY ASSIGNED IN THE FUTURE
NUM_OF_LABELS = 10

# NUMBER OF STEPS TO TRAIN
TRAIN_STEPS = 100

# PROPORTION OF EDGES TO SAMPLE
SAMPLE_CONST = 0.1

# ALPHAs - THESE ARE USED AS CONSTANTS IN MULTIPLICATION
ALPHA_1 = tf.constant(0.5, dtype=tf.float32, name="ALPHA_1")
ALPHA_2 = tf.constant(0.5, dtype=tf.float32, name="ALPHA_2")
ALPHA_3 = tf.constant(0.5, dtype=tf.float32, name="ALPHA_3")


def g_theta_total(index):
    """Produces a NUM_OF_LABELS by 1 vector describing the probs of each label

    Arguments:
        index {int} -- integer index of the specified node

    Returns:
        tf tensor -- tensor with the values of the probabilities
    """

    average_prob_value = np.zeros((NUM_OF_LABELS, 1))
    for neighbors in NODE_CONNECTIONS[index]:
        if LABEL_LIST.loc[neighbors, "LABELS"] == -1:
            pass
        else:
            average_prob_value[
                LABEL_LIST.loc[neighbors-1, "LABELS"]] += EDGE_MATRIX.loc[
                    index, neighbors]
    if sum(average_prob_value) == 0:
        return tf.convert_to_tensor(
            np.zeros((NUM_OF_LABELS, 1)) + 1/NUM_OF_LABELS)
    return tf.convert_to_tensor(average_prob_value/sum(average_prob_value))


def find_value(index, vector):
    co_ords = tf.where(tf.equal(index, vector), name="find_value_fn")[0][0]
    return vector[co_ords]


def d_term_h_index(pairs, nn_output):
    print("input shape %s" % pairs.get_shape())
    norm_vector = tf.map_fn(
        lambda a: tf.norm(
            nn_output[tf.to_int32(a[0])] - nn_output[tf.to_int32(a[0])]),
        tf.transpose(pairs))
    norm_vector = tf.to_float(tf.convert_to_tensor(norm_vector))
    # printed_norm = tf.Print(norm_vector, [norm_vector], message="THE NORM")
    print("Norm vector shape %s" % norm_vector.get_shape())
    return norm_vector


def get_neighbors(index):
    """Returns number of neighbors this node has

    Arguments:
        index {int} -- Node index

    Returns:
        int -- Number of nodes specified node is connected to
    """

    return tf.convert_to_tensor(np.count_nonzero(EDGE_MATRIX.loc[index]),
                                dtype=tf.float32)


def c_x(index, labels):
    """This function finds the cross entropy described in the loss fn

    Note: we do not need to check for erroneous labels (i.e. -1) bc they
    are already segmented into the three groups
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
    return tf.convert_to_tensor((1/get_neighbors(index)), dtype=tf.float32) * (
        tf.reduce_sum(
                  tf.one_hot(labels, depth=NUM_OF_LABELS, dtype=tf.float32) *
                  tf.log(tf.to_float(g_theta_total(index)))))


def custom_loss(labels, predicted, reference_vector, label_type_list):
    """The custom loss function for the NN

    Arguments:
        labels {tf tensor} -- list of labels
        predicted {tf tensor} -- list of predicted labels
        label_type_list {pandas DF} -- DF with all of the edge types

    Returns:
        tf tensor -- scalar of the final loss value
    """

    with tf.variable_scope('Loss') as loss_scope:

        LL_to_sample = round(len(label_type_list[0]) * SAMPLE_CONST)
        LU_to_sample = round(len(label_type_list[1]) * SAMPLE_CONST)
        UU_to_sample = round(len(label_type_list[2]) * SAMPLE_CONST)

        if LL_to_sample > 0:
            label_type_list[0] = [label_type_list[0][x]
                                  for x in np.random.randint(
                                    0, len(label_type_list[0]), LL_to_sample)]

        if LU_to_sample > 0:
            label_type_list[1] = [label_type_list[1][x]
                                  for x in np.random.randint(
                                    0, len(label_type_list[1]), LU_to_sample)]

        if UU_to_sample > 0:
            label_type_list[2] = [label_type_list[2][x]
                                  for x in np.random.randint(
                                    0, len(label_type_list[2]), UU_to_sample)]

        # iterate through each type of edge
        with tf.variable_scope('Labeled_edges', reuse=True) as scope:
            if label_type_list[0]:
                weight_tensor = tf.expand_dims(tf.convert_to_tensor(
                    [EDGE_MATRIX.loc[u_w, v_w]
                     for u_w, v_w in label_type_list[1]]), 1)
                label_tensor = tf.reshape(label_type_list[1], [-1])
                relative_indices = tf.map_fn(
                    lambda a: tf.to_float(
                        find_value(tf.to_int32(a), reference_vector)),
                    tf.to_float(label_tensor))
                relative_indices = tf.reshape(relative_indices, [2, -1])
                norm_tensor = tf.expand_dims(
                    d_term_h_index(relative_indices, predicted), axis=0)
                weight_norm_product = tf.reshape(
                    tf.matmul(norm_tensor, weight_tensor), [])
                c_uv_summed_term = tf.reduce_sum(
                    tf.convert_to_tensor(
                        [c_x(u_c, labels[u_c]) + c_x(v_c, labels[v_c])
                         for u_c, v_c in label_type_list[0]]))

                temp_sum_LL = ALPHA_1 * (
                    weight_norm_product + c_uv_summed_term)
                tf.summary.scalar("Labeled_subloss", temp_sum_LL)
            else:
                temp_sum_LL = 0

        with tf.variable_scope('Mixed_edges', reuse=True) as scope:
            if label_type_list[1]:
                weight_tensor = tf.expand_dims(tf.convert_to_tensor(
                    [EDGE_MATRIX.loc[u_w, v_w]
                     for u_w, v_w in label_type_list[1]]), 1)
                label_tensor = tf.reshape(label_type_list[1], [-1])
                relative_indices = tf.map_fn(
                    lambda a: tf.to_float(
                        find_value(tf.to_int32(a), reference_vector)),
                    tf.to_float(label_tensor))
                relative_indices = tf.reshape(relative_indices, [2, -1])
                norm_tensor = tf.expand_dims(
                    d_term_h_index(relative_indices, predicted), axis=0)
                weight_norm_product = tf.reshape(
                    tf.matmul(norm_tensor, weight_tensor), [])
                c_uv_summed_term = tf.reduce_sum(
                    tf.convert_to_tensor([c_x(u_c, labels[u_c])
                                          for u_c, v_c in label_type_list[0]]))
                temp_sum_LU = ALPHA_2 * (
                    weight_norm_product + c_uv_summed_term)
                tf.summary.scalar("Mixed_subloss", temp_sum_LU)
            else:
                temp_sum_LU = 0

        with tf.variable_scope('Unlabeled_edges', reuse=True) as scope:
            if label_type_list[2]:
                weight_tensor = tf.expand_dims(tf.convert_to_tensor(
                    [EDGE_MATRIX.loc[u_w, v_w]
                        for u_w, v_w in label_type_list[1]]), 1)
                label_tensor = tf.reshape(label_type_list[1], [-1])
                relative_indices = tf.map_fn(
                    lambda a: tf.to_float(
                        find_value(tf.to_int32(a), reference_vector)),
                    tf.to_float(label_tensor))
                relative_indices = tf.reshape(relative_indices, [2, -1])
                norm_tensor = tf.expand_dims(
                    d_term_h_index(relative_indices, predicted), axis=0)
                weight_norm_product = tf.reshape(
                    tf.matmul(norm_tensor, weight_tensor), [])
                temp_sum_UU = ALPHA_3 * weight_norm_product
                print(temp_sum_UU.get_shape())
                tf.summary.scalar("Unlabeled_subloss", (temp_sum_UU))
            else:
                temp_sum_UU = 0

        total_loss = tf.reduce_sum(
            tf.get_variable("Labeled_edges/temp_sum_LL", shape=[]) +
            tf.get_variable("Mixed_edges/temp_sum_LU", shape=[]) +
            tf.get_variable("Unlabeled_edges/temp_sum_UU", shape=[]))
        return total_loss


def make_dict_feature_col(dict_of_features):
    return [tf.feature_column.numeric_column(str(col))
            for col in dict_of_features]


def my_model_fn(dataset, hidden_nodes, log_dir):
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

    loss = custom_loss(dataset[2], logits, dataset[1], TOTAL_LLUU_LIST)
    optimizer = tf.train.GradientDescentOptimizer(0.01)
    train = optimizer.minimize(loss)

    init = tf.global_variables_initializer()

    with tf_debug.LocalCLIDebugWrapperSession(tf.Session()) as sess:
        writer = tf.summary.FileWriter("./tmp/log/"+log_dir, sess.graph)
        for var in tf.trainable_variables():
            tf.summary.histogram(var.name, var)
        all_summaries = tf.summary.merge_all()
        sess.run(init)
        for counter in range(100):
            _, loss_value = sess.run((train, loss))
            if counter % 5 == 0:
                summary = sess.run(all_summaries)
                writer.add_summary(summary, counter)
            print(loss_value)
        writer.close()

# THE DATASET IS COMPRISED OF INDEX VALUES TO IDENTIFY THE NODES,
# THE EDGE WEIGHTS, AND THE LABELS

# ------- ! BEGIN DATA IMPORT PIPELINE ! ------- #


zipped_features = {str(key): np.array(value)
                   for key, value in dict(EDGE_MATRIX).items()}

# THE DATASET'S FORMAT IS (FEATURE_ARRAY, SCALAR_INDEX, SCALAR_LABELS)
# THIS MEANS THAT ACCESSING THE DATASET'S FEATURES IS DATASET[0]

slices = tf.data.Dataset.from_tensor_slices(
    (zipped_features,
     np.arange(start=1, stop=501),
     LABEL_LIST.values.astype(int)))

# slices = slices.shuffle(100)
# slices = slices.batch(len(EDGE_MATRIX.index.values)).repeat(count=None)

slices = slices.batch(len(EDGE_MATRIX)).repeat(count=None)

# THIS WILL OUTPUT, ROW BY ROW, THE NODES
next_item = slices.make_one_shot_iterator().get_next()

# ------- ! END DATA IMPORT PIPELINE ! ------- #

my_model_fn(next_item, [500, 500, 20], "draft2")
