""" This module moves the creation of critical arrays
to here, instead of in random_walk.py, which now
will simply load the .txt files.
"""
import os
import sys
import numpy as np
import pandas as pd

FILENAME = "NN_NNMF_500Cves.txt"
IMPORT_ARRAY = np.loadtxt(os.path.join(sys.path[0], FILENAME), delimiter=' ')

NUM_OF_NODES = int(max(IMPORT_ARRAY[:, 0]))
NUM_OF_EDGES = IMPORT_ARRAY.shape[0]

# this gets a list of all nodes
NODE_LIST = set([int(node[0]) for node in IMPORT_ARRAY])
DEFAULT_NODES = set([default_node for default_node in range(NUM_OF_NODES+1)])
print("These nodes are missing")
print(DEFAULT_NODES-NODE_LIST)
print("This graph has %d nodes and %d edges" % (NUM_OF_NODES, NUM_OF_EDGES))

# This serves as the indices between the nodes
# The append is necessary as it maintains the format
INDEX_MARKERS = [index for index in range(IMPORT_ARRAY.shape[0])
                 if IMPORT_ARRAY[index][0] != IMPORT_ARRAY[index-1][0]]
INDEX_MARKERS.append(IMPORT_ARRAY.shape[0])

# this generates a list of lists - the first index is the node's index
# and the list inside signifies all of the nodes' connections
# DEPRECATED CODE THAT WAS A REALLY COOL LIST COMPREHENSION
# R I P :(
# NODE_CONNECTIONS = [[int(IMPORT_ARRAY[x][1]) for x in range(INDEX_MARKERS[y],
#                     INDEX_MARKERS[y+1])] for y in range(max(NODE_LIST))]
NODE_CONNECT_DEST = [
    [int(IMPORT_ARRAY[connection][0]) for connection in np.where(
     IMPORT_ARRAY[:, 1] == origin)[0]] for origin in NODE_LIST]

NODE_CONNECT_ORIG = [[int(IMPORT_ARRAY[x][1]) for x in range(INDEX_MARKERS[y],
                     INDEX_MARKERS[y+1])] for y in range(max(NODE_LIST))]

NODE_CONNECTIONS = [
    list(set(NODE_CONNECT_DEST[node_num-1] + NODE_CONNECT_ORIG[node_num-1]))
    for node_num in NODE_LIST]

NODE_CONNECTIONS.insert(0, [0])

TOTAL_WEIGHT_ARR = np.zeros((NUM_OF_NODES+1, NUM_OF_NODES+1))
for origin_node in IMPORT_ARRAY:
    TOTAL_WEIGHT_ARR[int(origin_node[0])][int(origin_node[1])] = origin_node[2]
    TOTAL_WEIGHT_ARR[int(origin_node[1])][int(origin_node[0])] = origin_node[2]

# create a list of nodes and their classifications
# the second dimension is wrt time
# ! LABEL_LIST needs to be assigned *REAL* labels!! :)
LABEL_LIST = np.zeros((NUM_OF_NODES+1, 1))-1
LABEL_LIST[4] = 1

# THE INDEX SHOULD BE NUMBERS FROM 1-500, VALUES ARE THE NODE CVE NAMES
LOOKUP_TABLE = pd.DataFrame(np.zeros([500, 1]), index=range(1, 501))


# FUNCTION TO BE REMOVED - - - OLD
def edge_type(node_1, node_2):
    if LABEL_LIST[node_1] != -1 and LABEL_LIST[node_2] != -1:
        # this is the LL case
        return 0
    elif LABEL_LIST[node_1] == -1 and LABEL_LIST[node_2] == -1:
        # this is the UU case
        return 2
    else:
        # this is the LU case
        return 1


def edge_type_finder(node_1, node_2, input_label_list):
    node_1_type = False
    node_2_type = False
    real_node_1_name = LOOKUP_TABLE[node_1]
    real_node_2_name = LOOKUP_TABLE[node_2]
    if real_node_1_name in input_label_list.index:
        node_1_type = True
    if real_node_2_name in input_label_list.index:
        node_2_type = True
    if node_1_type and node_2_type:
        # LL case
        return 0
    elif node_1_type != node_2_type:
        # LU case
        return 1
    else:
        # UU case
        return 2


def check_for_labeled_neighbors(node_index):
    for connections in NODE_CONNECTIONS[node_index]:
        if LABEL_LIST[connections][0] != -1:
            return True
    return False

NODE_TYPES = np.zeros((NUM_OF_NODES+1))-1
for node in range(1, NUM_OF_NODES+1):
    if LABEL_LIST[node][0] != -1:
        # SPECIFY AS LL
        NODE_TYPES[node] = 0
    elif check_for_labeled_neighbors(node):
        # SPECIFY AS LU
        NODE_TYPES[node] = 1
    else:
        # SPECIFY AS UU
        NODE_TYPES[node] = 2

PANDAS_WEIGHT_ARR = pd.DataFrame(
    TOTAL_WEIGHT_ARR[1:, 1:],
    index=NODE_LIST,
    columns=NODE_LIST)

PANDAS_NODE_LABELS = pd.DataFrame(
    {'label': LABEL_LIST[1:][0], 'type': NODE_TYPES[1:]},
    index=NODE_LIST,
    columns=["type"])


def create_edge_list(filename):
    TOTAL_LLUU_LIST = [[], [], []]
    imported_label_list = pd.from_pickle(filename)
    for row in IMPORT_ARRAY:
        TOTAL_LLUU_LIST[
            edge_type_finder(int(row[0]), int(row[1]),
                             imported_label_list)].append(
                                 [int(row[0]), int(row[1])])
    return TOTAL_LLUU_LIST


def write_to_disk(filename, list_to_write):
    """Writes a specific list to disk

    Arguments:
        filename {str} -- where to write the file
        list_to_write {list} -- list to write
    """

    with open(filename, 'w+') as filehandle:
        for listitem in list_to_write:
            filehandle.write('%s\n' % listitem)

write_to_disk('./data/node_list.txt', NODE_LIST)
write_to_disk('./data/nodes_missing.txt', DEFAULT_NODES - NODE_LIST)
write_to_disk('./data/label_list.txt', LABEL_LIST)
write_to_disk('./data/index_markers.txt', INDEX_MARKERS)
write_to_disk('./data/node_connections.txt', NODE_CONNECTIONS)
write_to_disk('./data/node_connect_dest.txt', NODE_CONNECT_DEST)
write_to_disk('./data/node_connect_orig.txt', NODE_CONNECT_ORIG)
np.savetxt("./data/edge_weights.txt", TOTAL_WEIGHT_ARR)
PANDAS_WEIGHT_ARR.to_pickle("./data/pandas_weight_array.pickle")

write_to_disk('./data/TET_arch.txt',
              create_edge_list("./data/architectural_concepts_cleaned_CWE"))
write_to_disk('./data/TET_dev.txt',
              create_edge_list("./data/development_concepts_cleaned_CWE"))
write_to_disk('./data/TET_res.txt',
              create_edge_list("./data/research_concepts_cleaned_CWE"))
