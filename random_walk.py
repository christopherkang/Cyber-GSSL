""" This module serves to act as a host for label propagation
General formatting note: often, matrices will have a size of
NUM_OF_NODES+1 - the 0th index is often filled with -1 or 0. This
is simply for convenience - when nodes are called, their names can
simply be used, instead of their index (which would be #-1)
"""
import os
import random
import numpy as np


def access_file(filename):
    """Accesses a given file

    Arguments:
        filename {str} -- file directory
    """
    temp_list = []
    if os.path.exists("/home/el/myfile.txt"):
        raise Exception('FileNotFound_%s' % (filename))
    with open(str(filename), "r") as filehandle:
        for line in filehandle:
            current_place = line[:-1]
            temp_list.append(eval(current_place))
        return temp_list

# This is the max number of iterations the file should run
MAX_ITERS = 100

# create a list of nodes and their classifications
# the second dimension is wrt time
# ! LABEL_LIST needs to be assigned *REAL* labels!! :)
LABEL_LIST = access_file("./data/label_list.txt")
LABEL_LIST = np.asarray(LABEL_LIST)
# This serves as the indices between the nodes
# The append is necessary as it maintains the format
INDEX_MARKERS = access_file("./data/index_markers.txt")

# this generates a list of lists - the first index is the node's index
# and the list inside signifies all of the nodes' connections
NODE_CONNECTIONS = access_file("./data/node_connections.txt")

# this gets a list of all nodes
NODE_LIST = access_file("./data/node_list.txt")
NUM_OF_NODES = int(max(NODE_LIST))
NUM_OF_EDGES = int(max(INDEX_MARKERS))

print("These nodes are missing")
print(access_file("./data/nodes_missing.txt"))
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


def check_neighbor(inp_node, connection_list, time_label_list):
    """Checks the neighbors of a node and outputs the most common labels.
    If there is a tie, the label is randomly chosen.
    This is one of the propagative functions.

    Arguments:
        inp_node {int} -- integer describing the index of the node
        connection_list {nested lists} -- defined in node_connections
        time_label_list {list} -- list with each node index assigned to a label

    Returns:
        int -- recommended label for that node.
    """
    temp_label_hold = []
    for neighbors in connection_list[inp_node]:
        if time_label_list[neighbors] == -1:  # only count "real" labels
            pass
        else:
            temp_label_hold.append(time_label_list[neighbors])
    # only return most common labels
    print(temp_label_hold)
    c = temp_label_hold.count  # shorting
    top_labels = list({x for x in temp_label_hold if c(x) ==
                       max(map(c, temp_label_hold))})
    print("these are in top_labels ", top_labels)
    # a set of the most popular labels
    if not top_labels:
        return -1
    return random.choice(top_labels)  # this is the label


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
    assert callable(propagating_function)
    node_order = shuffle_list(inp_nodes)
    time_counter = inp_labels.shape[1]  # this is the current time
    labels_after_time = np.concatenate(
        (inp_labels, np.zeros([NUM_OF_NODES + 1, 1])-1), axis=1)

    for node in node_order:  # for each node index
        labels_after_time[node][time_counter] = propagating_function(
            node, connection_list, inp_labels[:, -1])

    return labels_after_time

for counter in range(MAX_ITERS):
    LABEL_LIST = iterate_time(LABEL_LIST, NODE_LIST,
                              NODE_CONNECTIONS, check_neighbor)
print(LABEL_LIST)
