"""This file holds the weighted random walk (aka check_neighbor) algorithm
It factors the edge weight when evaluating the labels.
"""
import random
import numpy as np


def weight_check_neighbor(inp_node, connection_list, time_label_list, weights):
    """Checks the neighbors of a node and outputs the most common labels.
    If there is a tie, the label is randomly chosen.
    This is one of the propagative functions.

    Arguments:
        inp_node {int} -- integer describing the index of the node
        connection_list {nested lists} -- defined in node_connections
        time_label_list {list} -- list with each node index assigned to a label
        weights {dict} -- dict of weights, keys are dest nodes

    Returns:
        int -- recommended label for that node.
    """
    temp_label_hold = []
    sum_weights = {}
    for neighbors in connection_list[inp_node]:
        if time_label_list[neighbors] == -1:  # only count "real" labels
            pass
        else:
            try:
                sum_weights[time_label_list[neighbors]] += weights[neighbors]
            except:
                sum_weights[time_label_list[neighbors]] = weights[neighbors]
    try:
        max_value = max(sum_weights.values())
        max_value = list({key for key, value in sum_weights.items()
                          if value == max_value})
    except:
        return -1
    print("these are in top_labels ", max_value)
    # a set of the most popular labels
    return random.choice(max_value)  # this is the label
