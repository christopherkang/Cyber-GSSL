""" This module serves to act as a host for label propagation
General formatting note: often, matrices will have a size of
NUM_OF_NODES+1 - the 0th index is often filled with -1 or 0. This
is simply for convenience - when nodes are called, their names can
simply be used, instead of their index (which would be #-1)
"""
import numpy as np
from prop_algs import *

# This is the max number of iterations the file should run
MAX_ITERS = 1

# create a list of nodes and their classifications
# the second dimension is wrt time
# ! LABEL_LIST needs to be assigned *REAL* labels!! :)
LABEL_LIST = readfile.access_file("./data/label_list.txt")
LABEL_LIST = np.asarray(LABEL_LIST)

# This serves as the indices between the nodes
# The append is necessary as it maintains the format
INDEX_MARKERS = readfile.access_file("./data/index_markers.txt")

# this generates a list of lists - the first index is the node's index
# and the list inside signifies all of the nodes' connections
NODE_CONNECTIONS = readfile.access_file("./data/node_connections.txt")

# this gets a list of all nodes
NODE_LIST = readfile.access_file("./data/node_list.txt")
NUM_OF_NODES = int(max(NODE_LIST))
NUM_OF_EDGES = int(max(INDEX_MARKERS))

WEIGHT_MATRIX = np.loadtxt("./data/edge_weights.txt")

print("These nodes are missing")
print(readfile.access_file("./data/nodes_missing.txt"))
print("This graph has %d nodes and %d edges" % (NUM_OF_NODES, NUM_OF_EDGES))


def shuffle_list(inp_list):
    """Automatically shuffles a given list, nondestructively

    Arguments:
        inp_list {List} -- input list

    Returns:
        list -- shuffled list
    """
    temp_list = inp_list
    np.random.shuffle(temp_list)
    return temp_list


def predict_node(node_num, inp_labels, func_to_use):
    """Serves as function to pass

    Arguments:
        node_num {int} -- number of node to predict
        inp_labels {list} -- full list of all labels
        func_to_use {str} -- name of function to use
    """
    if func_to_use == "rand_walk":
        return rand_walk.check_neighbor(
            node_num, NODE_CONNECTIONS, inp_labels[:, -1])
    elif func_to_use == "weighted_rand_walk":
        weights = {
            dest_node: WEIGHT_MATRIX[node_num][dest_node] for dest_node in
            NODE_CONNECTIONS[node_num]}
        print(weights)
        return weighted_rand_walk.weight_check_neighbor(
            node_num, NODE_CONNECTIONS, inp_labels[:, -1], weights)
    elif func_to_use == "nge":
        raise Exception("NGECurrentlyUnsupported")
    else:
        raise Exception("UnknownFunctionError")


def iterate_time(inp_labels, inp_nodes, connection_list, propagating_function):
    """Iterator for time

    Arguments:
        inp_labels {matrix} -- matrix with all current/past labels;
        [:-1] is most recent
        inp_nodes {list} -- list of nodes that need to be calculated
        connection_list {nested lists} -- defined in node_connections
        propagating_function {function} -- propagating function used

    Returns:
        matrix -- new set of labels, should be one more deep
    """

    # check to insure the labels given match the expected shape
    assert inp_labels.shape[0] == NUM_OF_NODES+1
    node_order = shuffle_list(inp_nodes)
    time_counter = inp_labels.shape[1]  # this is the current time
    labels_after_time = np.concatenate(
        (inp_labels, np.zeros([NUM_OF_NODES + 1, 1])-1), axis=1)

    for node in node_order:  # for each node index
        labels_after_time[node][time_counter] = predict_node(
            node, inp_labels, propagating_function)

    return labels_after_time

for counter in range(MAX_ITERS):
    LABEL_LIST = iterate_time(LABEL_LIST, NODE_LIST,
                              NODE_CONNECTIONS, "weighted_rand_walk")
print(LABEL_LIST)
