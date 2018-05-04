#!/usr/bin/python3


#######
#
# Imports a graph, runs the SciPy Floyd-Warshall
# algorithm on it, timing the run, writes the
# time to a file, and then repeats.
#
# Note 1: This is set up for batch processing.
#         If you wish to run -just one-, then
#         have graphList contain the # of vertices
#         for the desired graph.
#
# Note 2: The input files are the Matrix Market
#         files produced by the 'generate_graph.py'
#         script.
#
# Note 3: You will need to manually change it from
#         degree 6 to degree 100 (or whatever degree
#         your graph is).
#
#######

import time

import numpy as np
import scipy.io as sio
from scipy.sparse import csc_matrix
from scipy.sparse.csgraph import floyd_warshall

#######
#
# Command line arguments - none
#
#######

# create list of graph vertices sizes
graphList = [1000, 2000, 3000, 5000, 7000, 10000, 15000, 20000, 25000, 30000]

# mainly for documentation, declare the filename for the graph file
filename = ""

# create variable for output filename
outputFile = "APSP_report.txt"

# loop over the list
for i in graphList:

    # zero out the inFile
    filename = ""

    # create the new one
    filename = str(i) + "-vertices_degree-6.mtx"

    # read graph into a SciPy sparse graph
    A = csc_matrix(sio.mmread(filename), dtype=np.int32)

    # get the start time
    start_time = time.time()

    # run the APSP algorithm on the graph
    path = floyd_warshall(A)

    # get the stop time and compute the difference
    elapsed_time = time.time() - start_time

    # open the report file in APPEND mode
    outFile = open(outputFile, "a")

    # write the runtime for the graph to the file
    outFile.write("Runtime for %s is %.6f ms\n\n" % (filename, (elapsed_time*1000)))

    # close the file
    outFile.close()

### end for 


