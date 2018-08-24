"""This file holds the TF algorithm used for the "neural graph engine"
"""
import os
import random

import pandas as pd
import tensorflow as tf

import readfile

# EDGE_MATRIX format: columns are connections, rows are individual nodes
# values are weights
os.chdir(os.path.dirname(__file__))
EDGE_MATRIX = pd.read_pickle("../data/pandas_weight_array.pickle")

NODE_CONNECTIONS = readfile.access_file("../data/node_connections.txt")
TOTAL_LLUU_LIST = readfile.access_file("../data/total_edge_type.txt")
LIST_OF_LABELED_EDGES = TOTAL_LLUU_LIST[0]
LIST_OF_MIXED_EDGES = TOTAL_LLUU_LIST[1]
LIST_OF_UNLABELED_EDGES = TOTAL_LLUU_LIST[2]

NUM_OF_LABELS = 10

# LABEL_LIST format: columns are labels and LL/LU/UU status
# rows are individual notes
LABEL_LIST = pd.read_pickle("I DONT KNOW THE FILE PATH")
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


def split_data(features, labels):

    return train_set, train_labels, valid_set, valid_labels


# used to create the feature columns necessary for the estimator
def make_feature_columns(features, index=True):
    """Creates feature columns for an estimator

    Arguments:
        features {pd df} -- dataframe with the entire matrix connection list

    Returns:
        set -- set of tf feature columns (numeric)
    """
    feature_columns = set()
    if index:
        feature_columns += tf.feature_column.numeric_column(
            features.index.values)
    feature_columns += set(tf.feature_column.numeric_column(feature)
                           for feature in features)
    return feature_columns


# input function
def my_input_fn(features, labels, batch_size=1, shuffle=True, num_epochs=None):
    """Serves as the tensorflow input function

    Arguments:
        features {pd df} -- pandas datafram
        labels {pd df} -- pandas dataframe

    Keyword Arguments:
        batch_size {int} -- size of the batches (default: {1})
        shuffle {bool} -- whether shuffling is enabled (default: {True})
        num_epochs {int} -- total number of times to batch (default: {None})

    Returns:
        tf dataset -- dataset of the data
    """

    # convert inputs into a dataset
    features = {key: (np.array(value), index) for
                key, value, index in
                zip(dict(features).items(), features.index.values)}
    ds = tf.data.Dataset.from_tensor_slices(features)

    # This will supply them indefinitely
    ds = ds.batch(batch_size).repeat(num_epochs)

    # shuffle
    if shuffle:
        ds = ds.shuffle(50)

    # return batched info
    features, labels = ds.make_one_shot_iterator().get_next()
    return features, labels


# need to write the model function
def my_model_fn(features, labels, mode, params):

    # the feature column input layer
    # applies feature columns to the data dict
    r_features, indices = tf.split(
        features, [EDGE_MATRIX.shape[0], 1], 1, name="Index_Feature_Split")
    
    net = tf.feature_column.input_layer(r_features, params['feature_columns'])
    for units in params['hidden_units']:
        # then, pass the output through the hidden layers
        net = tf.layers.dense(net, units=units, activation=tf.nn.relu)

    # the tf.nn.softmax can't be used as an activation function, so
    # it is applied afterwords
    logits = tf.layers.dense(
        net, params['n_classes'], activation=tf.nn.softmax)

    # Predict
    predicted_classes = tf.argmax(logits, 1)
    if mode == tf.estimator.ModeKeys.PREDICT:
        predictions = {
            'class_ids': predicted_classes[:, tf.newaxis],
            'probabilities': tf.nn.softmax(logits),
            'logits': logits
        }
        return tf.estimator.EstimatorSpec(mode, predictions=predictions)

    # Let's calculate loSS!!
    # THIS LOSS NEEDS TO BE FIXED
    # loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)

    match_output = tf.concat([indices, logits], 1)

    # Compute evaluation metrics
    accuracy = tf.metrics.accuracy(
        labels=labels, predictions=predicted_classes, name="acc_op")

    metrics = {'accuracy': accuracy}
    tf.summary.scalar('accuracy', accuracy[1])

    if mode == tf.estimator.ModeKeys.EVAL:
        return tf.estimator.EstimatorSpec(
            mode, loss=loss, eval_metric_ops=metrics)

    a_trainer = tf.train.AdagradOptimizer(0.1)
    train_op = a_trainer.minimize(loss, global_step=tf.train.get_global_step())
    return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=train_op)


training_input_fn = lambda: my_input_fn(FILL_ME_IN)
validation_input_fn = lambda: my_input_fn(FILL_ME_IN)
test_input_fn = lambda: my_input_fn(FILL_ME_IN)

classifier = tf.estimator.Estimator(
    model_fn=my_model_fn,
    params={'feature_columns': make_feature_columns(EDGE_MATRIX),
            'hidden_units': [1000, 500],
            'n_classes': 3,
            })

classifier.train(
    input_fn=training_input_fn,
    steps=TRAIN_STEPS)

classifier.predict(
    input_fn=validation_input_fn)

classifier.predict(
    input_fn=test_input_fn)
