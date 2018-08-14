"""This file holds the random walk (aka check_neighbor) algorithm
"""
import random


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


