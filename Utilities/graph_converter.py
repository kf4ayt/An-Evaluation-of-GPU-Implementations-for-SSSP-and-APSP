#!/usr/bin/python3

import sys

import numpy as np
import scipy.io as sio
from scipy.sparse import coo_matrix

import networkx as nx


#######
#
# The purpose of this program is to convert between
# the generated matrix market files and the proprietary
# formats used by the various paper authors
#
#######

#######
#
# Command line arguments
#
# sys.argv[0] - program name
# sys.argv[1] - type (for output)
# sys.argv[2] - filename (for input)
#
# type 1 = matrix market graph to be imported and converted to Harish & Yarayanan
# type 2 = matrix market graph to be imported and converted to Ortega
# type 3 = matrix market graph to be imported and converted to Martin et al (CUDA Solutions...)
#
#######


#######
### Get started
#######

G = nx.Graph()  # create an empty graph


#######
###
### If we're importing a market matrix file, regardless of the output,
### we need to bring it in, store it in a SciPy sparse graph, and then
### convert it to a NetworkX graph that is stored in the empty graph
### that we just created
###
#######

# read graph into a SciPy sparse graph
A = coo_matrix(sio.mmread(sys.argv[2]), dtype=np.int32)

# convert it to a networkx graph
G = nx.from_scipy_sparse_matrix(A)


#######
###
### Get the filename set up
###
#######

# Store the filename string from the cmd line in temp
temp = sys.argv[2]

# Remove the '.mtx' and store it in fileName. This will allow
# us to easily tack on an ending that will indicate the format

fileName = temp[:-4]


#######
#
# Now to convert
#
#######


# Type 1 - Harish & Yaranayan output format
#
if (sys.argv[1] == '1'):

    # open a file 
    thisFileName = fileName + ".Accel.txt"
    graphFile = open(thisFileName, "w")

    # print the number of vertices
    graphFile.write('%d\n' % nx.Graph.order(G))      # prints to the file

    ### DON'T skip a line!

    # Print the vertices in the format 'vertex degree'

    position = 0    # position in the vertex array to store the vertex degree

    for i in nx.nodes(G):
        graphFile.write('%d %d\n' % (position, nx.degree(G,i)))
        position += nx.degree(G,i)

    # NOW we print a blank line
    graphFile.write('\n')                   # prints a blank line to the file

    # Print the source vertex, followed by an empty line
    # For this, we will always use 0 as the source vertex
    #
    graphFile.write('0\n\n')                   # prints a blank line to the file

    ### Now to print the edge info
	
    # first, print the total number of edges
    #
    # Note: for this output format, we count the degree of each node
    #       to get the # of edges, not the actual num of edges. As it
    #       happens, 'position', after the above loop, has the desired
    #       value.

    graphFile.write('%d\n' % position)

    ### DON'T skip a line!

    # Print each node's edges and weights, it being in the form of:
    #   edge weight
    #
    # No lines are skipped between nodes

    for i in nx.nodes(G):           # we're going to go through each node
        for j in list(G[i].keys()):              # we're going to go through the edges

            graphFile.write('%d %d\n' % (j, G[i][j]['weight']))

            # All edges have to have a weight of at least 1
            if (G[i][j]['weight'] == 0):
                print('vertex %d, edge %d has weight 0' % (i, j))
                sys.exit()

    # Close the output file
    graphFile.close()


### end of if


# Type 2 - Ortega output format
#
if (sys.argv[1] == '2'):

    # open a file
    thatFileName = fileName + ".Ortega.txt"
    graphFile = open(thatFileName, "w")

    # start off by writing the number of vertices and the number of
    # edges on the first line in the format of 'num_vertices num_edges'

    # print the number of vertices
    graphFile.write('%d %d\n' % (nx.Graph.order(G), G.number_of_edges()))  # prints to the file
	
    ### DON'T print a blank line

    # print all of the vertices' position in an array, printing
    # them in one long string, all on one line
	
    vertexArray = '0'
    vertexNumArray = 0

    for i in nx.nodes(G):           # we're going to go through each node
        if i==0:
            vertexNumArray += nx.degree(G,i)
            # leave the vertexArray alone
            
        else:
            vertexArray = vertexArray + ' ' + str(vertexNumArray)        
            vertexNumArray += nx.degree(G,i)
    # end of for loop        

    # write the vertexArray to the output file and skip a line
    graphFile.write(vertexArray + '\n')
    graphFile.write('\n')                   # prints a blank line to the file

    # Now we write out each node's edge numbers, 1 per line
    
    edgeList = ''
    
    for i in nx.nodes(G):
        edgeList = ''
        
        for j in list(G[i].keys()):  # G[i][j] should be the edge #
            if (edgeList == ''):
                edgeList = str(j)
            else:
                edgeList = edgeList + ' ' + str(j)

        graphFile.write(edgeList + '\n')
	    
    # end of for loop
    
    # Now we skip a line
    graphFile.write('\n')
    
    # Now we repeat the edgeList, but for the weights
    weightList = ''
    
    for i in nx.nodes(G):
        weightList = ''
        
        for j in list(G[i].keys()):  # G[i][j] should be the edge #
            if (weightList == ''):
                weightList = str(G[i][j]['weight'])
            else:
                weightList = edgeList + ' ' + str(G[i][j]['weight'])

        graphFile.write(weightList)    # notice the lack of a newline character!
	    
    # end of for loop
    
    # close the output file
    graphFile.close()


### end of if


# Type 3 - Martin et al output format
#
if (sys.argv[1] == '3'):

    # open a file
    thatFileName = fileName + ".Martin.gr"
    graphFile = open(thatFileName, "w")

    # start off by writing the number of vertices and the number of
    # edges on the first line in the format of 'num_vertices num_edges'
    #
    # Note: For this format, we need to multiply the # of edges by 2

    # print the number of vertices
    graphFile.write('%d %d\n' % (nx.Graph.order(G), (G.number_of_edges()*2)))  # prints to the file
	
    ### Print a blank line
    graphFile.write('\n')  # prints to the file

    # print all of the vertices' position in an array, printing
    # them on one line at a time
	
    vertexNumArray = 0

    for i in nx.nodes(G):           # we're going to go through each node
        if i==0:
            graphFile.write('0\n')
            vertexNumArray += nx.degree(G,i)
        else:
            graphFile.write('%d\n' % vertexNumArray)        
            vertexNumArray += nx.degree(G,i)
    # end of for loop


    ### Skip a line
    graphFile.write('\n')                   # prints a blank line to the file

    # Now we write out each node's edge numbers, 1 per line
    
    for i in nx.nodes(G):

        for key in G[i].keys():  # G[i].keys() should be the edge #
            graphFile.write('%d\n' % key)

    # end of for loop
    
    # Now we skip a line
    graphFile.write('\n')
    
    # Now we repeat the edgeList, but for the weights
    
    for i in nx.nodes(G):
        
        for j in list(G[i].keys()):  # G[i][j] should be the edge #
            graphFile.write('%d\n' % G[i][j]['weight'])

    # end of for loop
    
    # close the output file
    graphFile.close()


### end of if


print()
print("Done")
print()


