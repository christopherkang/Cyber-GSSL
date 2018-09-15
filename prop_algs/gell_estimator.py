"""GRAPH ENGINE: LOW LEVEL (GELL)
"""

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import KFold

import os
import readfile

# THIS IS NECESSARY FOR WINDOWS SYSTEMS
os.chdir(os.path.dirname(__file__))
np.random.seed(0)

# EDGE_MATRIX format: columns are connections, rows are individual nodes
# values are weights
EDGE_MATRIX = pd.read_pickle("../data/pandas_weight_array.pickle")
EDGE_MATRIX[EDGE_MATRIX[EDGE_MATRIX<=0.01]>0] = 0.01
EDGE_MATRIX[EDGE_MATRIX>=0.99] = 0.99
print(EDGE_MATRIX.max().max())
assert EDGE_MATRIX.equals(EDGE_MATRIX.transpose())
assert EDGE_MATRIX.loc[5,5] == 0
assert EDGE_MATRIX.isnull().sum().sum() == 0

EDGE_MATRIX = EDGE_MATRIX.astype(np.float32)

# EDGE_MATRIX = pd.DataFrame(np.zeros((500, 500))+0.5, index=np.arange(1,501), columns=np.arange(1,501))

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
"""
LABEL_LIST = pd.DataFrame(
    np.zeros(500)-1, index=range(1, 501), columns=["LABELS"])
LABEL_LIST.loc[4, "LABELS"] = 1
LABEL_LIST = LABEL_LIST.astype(int)

LABEL_LIST = pd.DataFrame(
    np.arange(0, 500), index=range(1,501), columns=["LABELS"]
)
LABEL_LIST = LABEL_LIST.astype(int)
"""
LABEL_LIST = pd.DataFrame(
    np.zeros(500)-1, index=range(1, 501), columns=["LABELS"]
)
LABEL_LIST = LABEL_LIST.astype('int32')
# NUMBER OF LABELS - THIS SHOULD BE DYNAMICALLY ASSIGNED IN THE FUTURE
"""
NUM_OF_LABELS = LABEL_LIST[LABEL_LIST>=0].loc[:, "LABELS"].nunique()
"""
NUM_OF_LABELS = 3

# THIS IS A NESTED LIST THAT DESCRIBES THE CONNECTIONS EACH NODE HAS
# BE MINDFUL THAT THE 0TH INDEX IS ASSOCIATED WITH AN IMAGINARY "0TH"
# NODE, AND THEREFORE HAS NO CONNECTIONS
NODE_CONNECTIONS = readfile.access_file("../data/node_connections.txt")

# THIS IS A NESTED LIST THAT DESCRIBES THE TYPE OF EACH EDGE
TOTAL_LLUU_LIST = readfile.access_file("../data/total_edge_type.txt")

# NUMBER OF STEPS TO TRAIN
TRAIN_STEPS = 100

# PROPORTION OF EDGES TO SAMPLE
SAMPLE_CONST = 0.1

# ALPHAs - THESE ARE USED AS CONSTANTS IN MULTIPLICATION
a1 = 0.5
a2 = 0.5
a3 = 0.5

# ENABLE CROSS VALIDATION
CROSS_VAL = False
NUM_OF_SPLITS = 3


def custom_loss(labels, predicted, reference_vector, feature_set, label_type_list):
    """The custom loss function for the NN

    Arguments:
        labels {tf tensor} -- list of labels
        predicted {tf tensor} -- list of predicted labels
        label_type_list {pandas DF} -- DF with all of the edge types

    Returns:
        tf tensor -- scalar of the final loss value
    """
    def find_value(value, vector):
        """This function finds the coordinates of a value in a vector

    Arguments:
        value {tf int 32} -- integer to be found
        vector {tf vect} -- vector to be searched

    Returns:
        tf int -- coordinate to find the value
    """
        print(f"The vector's shape is {vector.get_shape()}")
        # co_ords = tf.where(tf.equal(
        #     tf.to_int32(value), tf.to_int32(vector)), name="find_value_fn")[0, 0]
        co_ords = vector[value-1]
        return co_ords

    def d_term_h_index(pairs, nn_output):
        """Returns the norms of pairs of vector indices from the nn output

        Arguments:
            pairs {tf int/float} -- vector indices
            nn_output {matrix} -- list of all the outputs of the nn

        Returns:
            vector -- vector of all the pairwise norms
        """

        pairs = tf.transpose(pairs)
        print(pairs.get_shape())
        print(f'pairs type is {pairs.dtype}')
        pairs = tf.Print(pairs, [pairs], "PAIRS", summarize=20)

        nn_output = tf.Print(nn_output, [nn_output], "NN_OUT", summarize=20)
        print(f'nn_out type is {nn_output.dtype}')
        norm_vector = tf.map_fn(
            lambda a: tf.square(tf.norm(
                nn_output[(a[0])] - nn_output[(a[1])])), pairs, dtype=tf.float32)
        print(f'norm vector type is {norm_vector.dtype}')
        # norm_vector = tf.to_float(norm_vector)
        norm_vector = tf.Print(norm_vector, [norm_vector], "NORM_VEC", summarize=20)
        return norm_vector

    def c_x(index, feature_set, labels, predicted):
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

        print(index.get_shape())

        # num_of_neighbors = tf.convert_to_tensor(np.count_nonzero(
        #     EDGE_MATRIX.loc[index]), dtype=tf.float32)

        num_of_neighbors = tf.count_nonzero(feature_set[index], 1)

        # THIS FUNCTION IS RELATIVELY COMPLEX: LET'S BREAK IT DOWN
        # FIRST, WE ARE CONVERTING THE VALUE TO A TENSOR
        # THEN, WE USE MAP_FN TO APPLY GET_NEIGHBORS TO EACH ELEMENT IN INDEX
        # NEXT, WE NEED TO SUM OVER THE PRODUCT AND LABELS
        # USING ONE_HOT, WE CAN CREATE A ONE HOT PROB VECTOR WITH 1 AS TRUE
        # WE MULTIPLY!
        cross_entropy = -(1/num_of_neighbors) * tf.matmul(
                    tf.one_hot(labels, depth=NUM_OF_LABELS, dtype=tf.float32),
                    tf.transpose(tf.log(predicted[index]+1e-8)))
        cross_entropy = tf.Print(cross_entropy, [cross_entropy], "XENTROPY")
        return cross_entropy

    def main_subloss(label_types_to_iterate, ref_vec,
                     given_logits, given_var_scope, feature_set):
        """This is the main component of loss shared across LL/LU/UU

        Arguments:
            label_types_to_iterate {list} -- nested list of paired node indices
            to search through (part of LL/LU/ or UU)
            ref_vec {vec} -- reference vector - the index of the vector corresponds
            to the actual number, and the value corresponds to the tf index
            given_logits {vec} -- values of the logit outputs
            given_var_scope {str} -- name of the variable scope to use

        Returns:
            int -- returns the product of the norms and the edge weights
        """

        with tf.variable_scope(given_var_scope) as scope:
            weight_tensor = tf.expand_dims(tf.convert_to_tensor(
                            [feature_set[u_w-1][v_w-1]
                             for u_w, v_w in label_types_to_iterate]), 1)
            label_tensor = tf.reshape(label_types_to_iterate, [-1])
            print(f'REF_VEC dtype {ref_vec.dtype}')
            relative_indices = tf.map_fn(
                lambda a: ref_vec[a-1], label_tensor)
            print(f'relative_indices dtype {relative_indices.dtype}')
            relative_indices = tf.reshape(relative_indices, [2, -1])
            norm_tensor = tf.expand_dims(
                d_term_h_index(relative_indices, given_logits), axis=0)
            # norm_tensor = tf.square(norm_tensor)
            product_norm_weight = tf.reshape(
                tf.matmul(norm_tensor, weight_tensor), [])
            return product_norm_weight

    def labeled_subloss(given_edge_list, ref_vec, labels, predicted, feature_set):
        # iterate through each type of edge
        with tf.variable_scope('Labeled_edges', reuse=tf.AUTO_REUSE) as scope:
            temp_sum_LL = tf.get_variable("temp_sum_LL", shape=[])
            if given_edge_list:
                ALPHA_1 = tf.constant(a1, dtype=tf.float32, name="ALPHA_1")
                weight_norm_product = main_subloss(
                    given_edge_list, ref_vec,
                    predicted, 'Labeled_edges', feature_set)
                c_uv_summed_term = tf.reduce_sum(
                    tf.convert_to_tensor(
                        [c_x(u_c, feature_set, labels[find_value(u_c, ref_vec)], predicted) +
                         c_x(v_c, feature_set, labels[find_value(v_c, ref_vec)], predicted)
                         for u_c, v_c in given_edge_list]))
                temp_sum_LL = ALPHA_1 * (
                    weight_norm_product + c_uv_summed_term)
                tf.summary.scalar("Labeled_subloss", temp_sum_LL)
                return temp_sum_LL
            else:
                return tf.constant(0, dtype=tf.float32)

    def mixed_subloss(given_edge_list, ref_vec, labels, predicted, feature_set):
        with tf.variable_scope('Mixed_edges', reuse=tf.AUTO_REUSE) as scope:
            temp_sum_LU = tf.get_variable("temp_sum_LU", shape=[])
            if given_edge_list:
                ALPHA_2 = tf.constant(a2, dtype=tf.float32, name="ALPHA_2")
                weight_norm_product = main_subloss(
                    given_edge_list, ref_vec,
                    predicted, 'Mixed_edges', feature_set)
                c_uv_summed_term = tf.reduce_sum(
                    tf.convert_to_tensor(
                        [c_x(u_c, feature_set, labels[find_value(u_c, ref_vec)], predicted)
                            for u_c, u_v in given_edge_list]))
                temp_sum_LU = ALPHA_2 * (
                    weight_norm_product + c_uv_summed_term)
                tf.summary.scalar("Mixed_subloss", temp_sum_LU)
                return temp_sum_LU
            else:
                return tf.constant(0, dtype=tf.float32)

    def unlabeled_subloss(given_edge_list, ref_vec, labels, predicted, feature_set):
        with tf.variable_scope('Unlabeled_edges', reuse=tf.AUTO_REUSE) as scope:
            temp_sum_UU = tf.get_variable("temp_sum_UU", shape=[])
            if given_edge_list:
                ALPHA_3 = tf.constant(a3, dtype=tf.float32, name="ALPHA_3")
                weight_norm_product = main_subloss(
                    given_edge_list, ref_vec,
                    predicted, 'Unlabeled_edges', feature_set)
                temp_sum_UU = ALPHA_3 * weight_norm_product
                tf.summary.scalar("Unlabeled", temp_sum_UU)
                return temp_sum_UU
            else:
                return tf.constant(0, dtype=tf.float32)
    
    with tf.variable_scope('Loss') as loss_scope:
        total_loss = tf.convert_to_tensor(0)
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

        total_loss = (
            labeled_subloss(
                label_type_list[0], reference_vector, labels, predicted, feature_set) +
            mixed_subloss(
                label_type_list[1], reference_vector, labels, predicted, feature_set) +
            unlabeled_subloss(
                label_type_list[2], reference_vector, labels, predicted, feature_set)
        )
        tf.summary.scalar("LOSS", total_loss)
        return total_loss


def my_model_fn(features, labels, mode, params):
    """NN Model function

    Arguments:
        dataset {pd DF} -- dataset with indices
        hidden_nodes {list} -- list of hidden_nodes
    """

    # THIS CREATES THE INPUT LAYER AND IMPORTS DATA
    net = tf.feature_column.input_layer(
        features[0], params['feat_cols'])

    saved_features = net

    # BUILDS HIDDEN LAYERS
    for units in params['hidden_nodes']:
        net = tf.layers.dense(net, units=units, activation=tf.nn.relu)
        # kernel_initializer=tf.initializers.random_normal
    # BUILDS THE FINAL LAYERS
    logits = tf.layers.dense(
        net, params['classes'], activation=None)
    predicted_classes = tf.argmax(logits, 1)
    logits = tf.Print(logits, [logits], "LOGITS")


    scaled_logits = tf.nn.softmax(logits)
    # scaled_logits = tf.clip_by_value(scaled_logits, 1e-10, 1)
    print(logits.dtype)
    print(f'Ref Vec dtype{features[1].dtype}')
    loss = custom_loss(
        labels, scaled_logits, features[1], saved_features, params['LLUU_LIST'])

    loss = tf.Print(loss, [loss])

    if mode == tf.estimator.ModeKeys.PREDICT:
        predictions = {
            'predicted_classes': predicted_classes,
            'probabilities': scaled_logits,
            #'logits': logits,
            #'loss': loss,
        }
        return tf.estimator.EstimatorSpec(mode, predictions=predictions)

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

    # optimizer = tf.train.GradientDescentOptimizer(0.00000001)
    optimizer = tf.train.AdamOptimizer(0)
    # gvs = optimizer.compute_gradients(loss)
    # gvs = replace_none_with_zero(gvs)
    # print(gvs)
    # capped_gvs = [(tf.clip_by_value(grad, -1, 1), var) for grad, var in gvs]
    # train = optimizer.apply_gradients(capped_gvs)
    train = optimizer.minimize(loss, global_step=tf.train.get_global_step())
    for var in tf.trainable_variables():
        tf.summary.histogram(var.name, var)

    summary_hook = tf.train.SummarySaverHook(
        5,
        output_dir='/tmp/tf',
        summary_op=tf.summary.merge_all())

    return tf.estimator.EstimatorSpec(
        mode, loss=loss, train_op=train, training_hooks=[summary_hook])


def make_dict_feature_col(dict_of_features):
    return [tf.feature_column.numeric_column(str(col))
            for col in dict_of_features]


def input_fn(feature_set, label_list, shuffle=False):
    """Used to convert numpy arrays into tf Dataset slicese

    Arguments:
        feature_set {pandas df} -- 500x500 feature array
        label_list {pandas df} -- 500x1 labels

    Keyword Arguments:
        shuffle {bool} -- decides whether data should be shuffled
        (default: {False})

    Returns:
        tf Dataset iterator -- iterator to be used within model_fn
    """

    mapping_table = np.random.choice(len(feature_set), len(feature_set), replace=False)

    feature_set = feature_set.loc[mapping_table, :]

    zipped_features = {str(key): np.array(value)
                       for key, value in dict(feature_set).items()}
    # create the mapping table 

    slices = tf.data.Dataset.from_tensor_slices(
        ((zipped_features,
            mapping_table.astype(np.int32)),
            label_list.values.astype(np.float32)))
    if shuffle:
        slices = slices.shuffle(100)
    slices = slices.batch(len(feature_set)).repeat(count=None)
    return slices.make_one_shot_iterator().get_next()


def check_neighbors(node_index_1, node_index_2):
    """Determines the type of an edge

    Arguments:
        node_index_1 {int} -- index of arbitrary node
        node_index_2 {int} -- index of arbitrary node

    Returns:
        int -- this refers to the 1st level index of a nested list
    """

    if (LABEL_LIST.loc[node_index_1][0] != -1 and
       LABEL_LIST.loc[node_index_2][0] != -1):
        return 0
    elif (LABEL_LIST.loc[node_index_1][0] == -1 and
          LABEL_LIST.loc[node_index_2][0] == -1):
        return 2
    else:
        return 1


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
        classifier = tf.estimator.Estimator(
            model_fn=my_model_fn,
            model_dir="./tmp/log/draft7",
            params={"hidden_nodes": [500, 500, 20],
                    "log_dir": "draft6",
                    "LLUU_LIST": train_LLUU})

        classifier.train(
            input_fn=lambda: input_fn(train_feature, train_label), steps=10)

        # output_dict = classifier.predict()

        # loss_table.append(output_dict['loss'])
        # prediction_table.append(output_dict['predictions'])
else:
    train_LLUU = [[], [], []]
    for node_1 in EDGE_MATRIX.loc[1:500, 1:500].index.values:
        selected_row = EDGE_MATRIX.loc[node_1, 1:500]
        for node_2 in selected_row.nonzero()[0]:
            if node_2 > node_1:
                train_LLUU[check_neighbors(node_1,
                           EDGE_MATRIX.loc[1:500, 1:500].columns.values[node_2])].append(
                           [node_1, EDGE_MATRIX.loc[1:500, 1:500].columns.values[node_2]])
    assert not train_LLUU[0]
    assert not train_LLUU[1]
    my_feat_cols = []
    for feat_col in EDGE_MATRIX.loc[1:500, 1:500].index.values:
        my_feat_cols.append(tf.feature_column.numeric_column(str(feat_col)))
    classifier = tf.estimator.Estimator(
        model_fn=my_model_fn,
        model_dir="/tmp/log/newdrafts12",
        # model_dir="C:/Users/kang828/Desktop/pleasedeargodwork",
        params={"hidden_nodes": [30, 30, 30],
                'classes': NUM_OF_LABELS,
                "LLUU_LIST": train_LLUU,
                'feat_cols': my_feat_cols, })

    classifier.train(
        input_fn=lambda: input_fn(EDGE_MATRIX.loc[1:500, 1:500], LABEL_LIST.loc[1:500]), steps=1000)
    # predictions = classifier.predict(input_fn=lambda: input_fn(EDGE_MATRIX, LABEL_LIST))
    # for n in zip(predictions):
    #     print(n)
    # classifier.predict()
