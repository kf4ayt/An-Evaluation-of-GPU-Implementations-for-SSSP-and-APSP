
Name: graph_converter.py

Purpose: This script converts a Matrix Market graph to the desired
         file format for the SSSP/APSP implementations of Harish and
         Yarananan, Martin et al, and the format used by another paper
         whose code was ultimately not included in the project.

Requirements: This Python script requires Python 3 and the following
              packages: sys, numpy, scipy, networkx.

Usage: To use it, type './graph_converter.py' (be sure that it is an
       executable), the output format desired, and then the filename
       for the input graph. The numbers for the format are listed in
       the script. The output, using Martin et al as an example, for
       './graph_converter.py 3 abc.mtx' will be a file named
       'abc.Martin.gr'. Each format has its own extension, so as to
       differentiate between the various output files.


Charles Johnson
April, 2018


