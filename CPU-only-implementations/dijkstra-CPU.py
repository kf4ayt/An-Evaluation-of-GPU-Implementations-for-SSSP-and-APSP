#!/usr/bin/python3


#######
#
# Imports a graph and then runs Dijkstra's algorithm
# on it.
#
#######


import sys
import time

import numpy as np
import scipy.io as sio
from scipy.sparse import coo_matrix

import networkx as nx

#######
#
# Command line arguments
#
# sys.argv[0] - program name
# sys.argv[1] - filename for input
#
#######


#######
### Get started
#######

G = nx.Graph()  # create an empty graph


#######
###
### If we're importing a Matrix Market file, regardless of the output,
### we need to bring it in, store it in a SciPy sparse graph, and then
### convert it to a NetworkX graph that is stored in the empty graph
### that we just created
###
#######

# read graph into a SciPy sparse graph
A = coo_matrix(sio.mmread(sys.argv[1]), dtype=np.int32)

# convert it to a networkx graph
G = nx.from_scipy_sparse_matrix(A)


#######
#
# Now to run Dijkstra's on the graph
#
#######

print("Starting Dijkstra's...")

start_time = time.time()

path = nx.single_source_dijkstra_path(G, 0)

elasped_time = time.time() - start_time

print("%d vertices, %d edges" % (G.number_of_nodes(), G.number_of_edges()))

print("Function took %.4f seconds." % elasped_time)
print("Function took %.4f milliseconds." % (elasped_time*1000))
print()


