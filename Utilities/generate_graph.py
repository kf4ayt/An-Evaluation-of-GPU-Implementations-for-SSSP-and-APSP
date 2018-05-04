#!/usr/bin/python3

import random
import sys

import numpy as np
import scipy.io as sio

import networkx as nx


#######
#
# The purpose of this script is to generate a random
# graph of D degrees and V vertices. It will then
# save it to disk in a Matrix Market formatted file.
#
#######


#######
#
# Command line arguments
#
# sys.argv[0] - program name
# sys.argv[1] - number of vertices 
# sys.argv[2] - degree
#
#######


#######
#
# Get started
#
#######

G = nx.Graph()  # create an empty graph

# convert command line arguments to ints
num_vertices = int(sys.argv[1])
degree = int(sys.argv[2])

# formulate the output filename
outFile = str(num_vertices) + "-vertices_degree-" + str(degree) + ".mtx"


#######
#
# Create the random graph
#
#######

G = nx.random_regular_graph(int(degree), int(num_vertices))

# While this is time-consuming, we want to check and make sure that all
# nodes/vertices are of at least degree 1 (they should be whatever
# degree was specified, but...). If any are of degree 0, print the node
# and exit the program.

for v in nx.nodes(G):
    if (nx.degree(G, v) == 0):
        print()
        print('vertex %s is of degree: %d' % (v, nx.degree(G, v)))
        print()
        sys.exit()	# here's where we bail

# Now that we know that all nodes are OK, we need to add a weight
# to each edge. I've chosen to make the weights to be random ints
# between 1 and 10.

for (u, v) in G.edges():
    G[u][v]['weight'] = 0

    while (G[u][v]['weight'] == 0):
        G[u][v]['weight'] = random.randint(0,10)  # this might should be (1,10), but since it was
                                                  # like this for the tests, I'm not going to change it upon review 

#######
#
# Now, it is time to write it to disk.
#
#######

# convert the graph to a SciPy sparse matrix
A = nx.to_scipy_sparse_matrix(G)

# write the converted graph to a Matrix Market file
sio.mmwrite(outFile, A, field="integer", symmetry="general")


#######


print()
print("Done")
print()


