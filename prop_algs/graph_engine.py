"""This file holds the TF algorithm used for the "neural graph engine"
"""
import readfile
import tensorflow as tf
import pandas as pd

# EDGE_MATRIX format: columns are connections, rows are individual nodes
# values are weights
EDGE_MATRIX = pd.read_pickle("../data/pandas_weight_array.pickle")

# LABEL_LIST format: columns are labels and LL/LU/UU status
# rows are individual notes
LABEL_LIST = pd.read_pickle("I DONT KNOW THE FILE PATH")
TRAIN_STEPS = 100


def split_data(features, labels):

    return train_set, train_labels, valid_set, valid_labels


# used to create the feature columns necessary for the estimator
def make_feature_columns(features):
    """Creates feature columns for an estimator

    Arguments:
        features {pd df} -- dataframe with the entire matrix connection list

    Returns:
        set -- set of tf feature columns (numeric)
    """

    return set(tf.feature_column.numeric_column(feature)
               for feature in features)


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
    features = {key: np.array(value) for key, value in dict(features).items()}
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
    net = tf.feature_column.input_layer(features, params['feature_columns'])
    for units in params['hidden_units']:
        # then, pass the output through the hidden layers
        net = tf.layers.dense(net, units=units, activation=tf.nn.relu)

    # the tf.nn.softmax can't be used as an activation function, so
    # it is applied afterwords
    logits = tf.layers.dense(net, params['n_classes'], activation=None)

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
    loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)

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
