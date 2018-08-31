"""GRAPH ENGINE: LOW LEVEL (GELL)
"""

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.python import debug as tf_debug
from sklearn.model_selection import KFold

import os
import readfile

# THIS IS NECESSARY FOR WINDOWS SYSTEMS
os.chdir(os.path.dirname(__file__))
np.random.seed(0)

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
a1 = 0.5
a2 = 0.5
a3 = 0.5

# ENABLE CROSS VALIDATION
CROSS_VAL = True
NUM_OF_SPLITS = 3


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
                LABEL_LIST.loc[neighbors, "LABELS"]] += EDGE_MATRIX.loc[
                    index, neighbors]
    if sum(average_prob_value) == 0:
        return tf.convert_to_tensor(
            average_prob_value + 1/NUM_OF_LABELS)
    return tf.convert_to_tensor(average_prob_value/sum(average_prob_value))


def find_value(index, vector):
    co_ords = tf.where(tf.equal(
        tf.to_int32(index), tf.to_int32(vector)), name="find_value_fn")[0, 0]
    return co_ords


def d_term_h_index(pairs, nn_output):
    norm_vector = tf.map_fn(
        lambda a: tf.norm(
            nn_output[tf.to_int32(a[0])] - nn_output[tf.to_int32(a[1])]),
        tf.transpose(pairs))
    norm_vector = tf.to_float(tf.convert_to_tensor(norm_vector))
    return norm_vector


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

    num_of_neighbors = tf.convert_to_tensor(np.count_nonzero(
        EDGE_MATRIX.loc[index]), dtype=tf.float32)

    # THIS FUNCTION IS RELATIVELY COMPLEX: LET'S BREAK IT DOWN
    # FIRST, WE ARE CONVERTING THE VALUE TO A TENSOR
    # THEN, WE USE MAP_FN TO APPLY GET_NEIGHBORS TO EACH ELEMENT IN INDEX
    # NEXT, WE NEED TO SUM OVER THE PRODUCT AND LABELS
    # USING ONE_HOT, WE CAN CREATE A ONE HOT PROB VECTOR WITH 1 AS TRUE
    # WE MULTIPLY!
    return -tf.convert_to_tensor(
        (1/num_of_neighbors), dtype=tf.float32) * (tf.matmul(
                  tf.one_hot(labels, depth=NUM_OF_LABELS, dtype=tf.float32),
                  tf.log(tf.to_float(g_theta_total(index))+(10**-15))))


def main_subloss(label_types_to_iterate, ref_vec,
                 given_logits, given_var_scope):
    with tf.variable_scope(given_var_scope) as scope:
        weight_tensor = tf.expand_dims(tf.convert_to_tensor(
                        [EDGE_MATRIX.loc[u_w, v_w]
                         for u_w, v_w in label_types_to_iterate]), 1)
        label_tensor = tf.reshape(label_types_to_iterate, [-1])
        relative_indices = tf.map_fn(
            lambda a: tf.to_float(
                find_value(tf.to_int32(a), tf.to_int32(ref_vec))),
            tf.to_float(label_tensor))
        relative_indices = tf.reshape(relative_indices, [2, -1])
        norm_tensor = tf.expand_dims(
            d_term_h_index(relative_indices, given_logits), axis=0)
        norm_tensor = tf.square(norm_tensor)
        product_norm_weight = tf.reshape(
            tf.matmul(norm_tensor, weight_tensor), [])
        return product_norm_weight


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
                ALPHA_1 = tf.constant(a1, dtype=tf.float32, name="ALPHA_1")
                weight_norm_product = main_subloss(
                    label_type_list[0], reference_vector,
                    predicted, 'Labeled_edges')
                c_uv_summed_term = tf.reduce_sum(
                    tf.convert_to_tensor(
                        [c_x(u_c, labels[find_value(u_c, reference_vector)]) +
                         c_x(v_c, labels[find_value(v_c, reference_vector)])
                         for u_c, v_c in label_type_list[0]]))
                temp_sum_LL = ALPHA_1 * (
                    weight_norm_product + c_uv_summed_term)
                tf.summary.scalar("Labeled_subloss", temp_sum_LL)
            else:
                temp_sum_LL = 0

        with tf.variable_scope('Mixed_edges', reuse=True) as scope:
            if label_type_list[1]:
                ALPHA_2 = tf.constant(a2, dtype=tf.float32, name="ALPHA_2")
                weight_norm_product = main_subloss(
                    label_type_list[1], reference_vector,
                    predicted, 'Mixed_edges')
                c_uv_summed_term = tf.reduce_sum(
                    tf.convert_to_tensor(
                        [c_x(u_c, labels[find_value(u_c, reference_vector)])
                         for u_c, u_v in label_type_list[1]]))
                temp_sum_LU = ALPHA_2 * (
                    weight_norm_product + c_uv_summed_term)
                tf.summary.scalar("Mixed_subloss", temp_sum_LU)
            else:
                temp_sum_LU = 0

        with tf.variable_scope('Unlabeled_edges', reuse=True) as scope:
            if label_type_list[2]:
                ALPHA_3 = tf.constant(a3, dtype=tf.float32, name="ALPHA_3")
                weight_norm_product = main_subloss(
                    label_type_list[2], reference_vector,
                    predicted, 'Unlabeled_edges')
                temp_sum_UU = ALPHA_3 * weight_norm_product
                tf.summary.scalar("Unlabeled", temp_sum_UU)
            else:
                temp_sum_UU = 0

        total_loss = (
            tf.get_variable("Labeled_edges/temp_sum_LL", shape=[]) +
            tf.get_variable("Mixed_edges/temp_sum_LU", shape=[]) +
            tf.get_variable("Unlabeled_edges/temp_sum_UU", shape=[]))
        tf.summary.scalar("LOSS", total_loss)
        return total_loss


def make_dict_feature_col(dict_of_features):
    return [tf.feature_column.numeric_column(str(col))
            for col in dict_of_features]


def my_model_fn(features, labels, mode, params):
    """NN Model function

    Arguments:
        dataset {pd DF} -- dataset with indices
        hidden_nodes {list} -- list of hidden_nodes
    """

    # THIS CREATES THE INPUT LAYER AND IMPORTS DATA
    net = tf.feature_column.input_layer(
        features[0], make_dict_feature_col(features[0]))

    # BUILDS HIDDEN LAYERS
    for units in params['hidden_nodes']:
        net = tf.layers.dense(net, units=units, activation=tf.nn.relu)

    # BUILDS THE FINAL LAYERS
    logits = tf.layers.dense(
        net, NUM_OF_LABELS, activation=None)

    predicted_classes = tf.argmax(logits, 1)

    scaled_logits = tf.nn.softmax(logits)

    if mode == tf.estimator.ModeKeys.PREDICT:
        predictions = {
            'predicted_classes': predicted_classes,
            'probabilities': scaled_logits,
            'logits': logits,
        }
        return tf.estimator.EstimatorSpec(mode, predictions=predictions)

    loss = custom_loss(
        labels, scaled_logits, features[1], params['LLUU_LIST'])

    accuracy = tf.metrics.accuracy(
        labels=labels, predictions=predicted_classes, name='acc_op')

    metrics = {
        'accuracy': accuracy,
    }
    tf.summary.scalar('accuracy', accuracy[1])

    if mode == tf.estimator.ModeKeys.EVAL:
        return tf.estimator.EstimatorSpec(
            mode, loss=loss, eval_metric_ops=metrics
        )

    optimizer = tf.train.GradientDescentOptimizer(0.01)
    train = optimizer.minimize(loss, global_step=tf.train.get_global_step())
    for var in tf.trainable_variables():
        tf.summary.histogram(var.name, var)

    return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=train)


def input_fn(feature_set, label_list, shuffle=False):
    zipped_features = {str(key): np.array(value)
                       for key, value in dict(feature_set).items()}
    slices = tf.data.Dataset.from_tensor_slices(
        ((zipped_features,
            feature_set.index.values),
            label_list.values.astype(int)))
    if shuffle:
        slices = slices.shuffle(100)
    slices = slices.batch(len(feature_set)).repeat(count=None)
    slices.make_one_shot_iterator().get_next()
    return slices


def check_neighbors(node_index_1, node_index_2):
    if (LABEL_LIST.loc[node_index_1][0] != -1 and
       LABEL_LIST.loc[node_index_2][0] != -1):
        return 0
    elif (LABEL_LIST.loc[node_index_1][0] == -1 and
          LABEL_LIST.loc[node_index_2][0] == -1):
        return 2
    else:
        return 1

"""
if CROSS_VAL:
    loss_table = []
    prediction_table = []
    kf = KFold(n_splits=NUM_OF_SPLITS, shuffle=True)
    kf.get_n_splits(NUM_OF_SPLITS)
    for train, test in kf.split(EDGE_MATRIX):
        with tf.name_scope("Input_pipeline") as scope:
            train_feature = EDGE_MATRIX.iloc[train]
            train_label = LABEL_LIST.iloc[train]
            test_feature = EDGE_MATRIX.iloc[test]
            test_label = LABEL_LIST.iloc[test]

            train_dataset = input_fn(train_feature, train_label)
            test_dataset = input_fn(test_feature, test_label)

            edge_list = EDGE_MATRIX[
                EDGE_MATRIX.index.isin(train_feature.index.values)]
            edge_list = edge_list[
                edge_list.index.intersection(train_feature.index.values)]

            train_LLUU = [[], [], []]
            for node_1 in edge_list.index.values:
                selected_row = edge_list.loc[node_1]
                for node_2 in selected_row.nonzero()[0]:
                    if node_2 > node_1:
                        train_LLUU[check_neighbors(node_1,
                                   edge_list.columns.values[node_2])].append(
                                       [node_1,
                                        edge_list.columns.values[node_2]])

        losses, preds = my_model_fn(
            train_dataset, test_dataset, [500, 500, 20],
            "draft4_1", LLUU_LIST=train_LLUU)
        loss_table.append(losses)
        prediction_table.append(preds)
"""
classifier = tf.estimator.Estimator(
    model_fn=my_model_fn,
    model_dir="C:/Users/kang828/Desktop/Cyber-GSSL/prop_algs/tmp/log/draft6",
    params={"hidden_nodes": [500, 500, 20],
            "log_dir": "draft6",
            "LLUU_LIST": TOTAL_LLUU_LIST})

classifier.train(input_fn=lambda: input_fn(EDGE_MATRIX, LABEL_LIST), steps=10)
