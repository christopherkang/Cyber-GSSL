""" This module serves to act as a host for label propagation"""
import os
import sys
import random
import numpy as np

IN_FILENAME = "NN_NNMF_500Cves.txt" #set 
IMPORT_ARRAY = np.loadtxt(os.path.join(sys.path[0],IN_FILENAME), delimiter = ' ')

MAX_ITERS = 100

NUMBER_OF_NODES = int(max(IMPORT_ARRAY[:,0]))
NUMBER_OF_EDGES = IMPORT_ARRAY.shape[0]

#this gets a list of all nodes
NODE_LIST = set([int(x[0]) for x in IMPORT_ARRAY])
DEFAULT_NODE_LIST = set([x for x in range(NUMBER_OF_NODES+1)])
print("These nodes are missing")
print(DEFAULT_NODE_LIST-NODE_LIST)
print("This graph has %d nodes and %d edges"%(NUMBER_OF_NODES,NUMBER_OF_EDGES))

#create a list of nodes and their classifications
#the second dimension is wrt time
LABEL_LIST = np.zeros((NUMBER_OF_NODES+1,1))-1
LABEL_LIST[4] = 1

INDEX_MARKERS = [x for x in range(IMPORT_ARRAY.shape[0]) if IMPORT_ARRAY[x][0] != IMPORT_ARRAY[x-1][0]]
INDEX_MARKERS.append(IMPORT_ARRAY.shape[0])

#this generates a list of lists - the first index is the node's index
#and the list inside signifies all of the nodes' connections
NODE_CONNECTIONS = [[int(IMPORT_ARRAY[x][1]) for x in 
    range(INDEX_MARKERS[y],INDEX_MARKERS[y+1])]
    for y in range(max(NODE_LIST))]
NODE_CONNECTIONS.insert(0, 0)

def shuffle_list(inp_list):
    """Automatically shuffles a list, nondestructively
    
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
        if time_label_list[neighbors] == -1: #only count "real" labels
            pass
        else:
            temp_label_hold.append(time_label_list[neighbors])
    #only return most common labels
    print(temp_label_hold)
    c=temp_label_hold.count #shorting
    top_labels = list({x for x in temp_label_hold if c(x)==max(map(c,temp_label_hold))})
    print("these are in top_labels ",top_labels) 
    #a set of the most popular labels
    if len(top_labels) == 0:
        return -1
    else:
        return random.choice(top_labels) #this is the label

def iterate_time(inp_labels, inp_nodes, connection_list, propagating_function):
    """Iterator for time
    
    Arguments:
        inp_labels {matrix} -- matrix with all current/past labels, ([:-1] is most recent)
        inp_nodes {list} -- list of nodes that need to be calculated
        connection_list {nested lists} -- defined in node_connections
        propagating_function {function} -- propagating function used
    
    Returns:
        matrix -- new set of labels, should be one more deep
    """

    # check to insure the labels given match the expected shape 
    assert inp_labels.shape[0] == NUMBER_OF_NODES+1
    assert callable(propagating_function)
    node_order = shuffle_list(inp_nodes)
    time_counter = inp_labels.shape[1] #this is the current time
    labels_after_time = np.concatenate(
        (inp_labels, np.zeros([NUMBER_OF_NODES+1,1])-1), axis = 1)

    for node in node_order: #for each node index
        labels_after_time[node][time_counter] = propagating_function(
            node, connection_list, inp_labels[:,-1])

    return labels_after_time

for counter in range(MAX_ITERS):
    LABEL_LIST = iterate_time(LABEL_LIST,list(NODE_LIST),NODE_CONNECTIONS,check_neighbor)
print(LABEL_LIST)
