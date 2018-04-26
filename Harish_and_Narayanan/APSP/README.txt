
In order to compile this program, edit the Makefile so that the GPU architecture
is correct (in my case, I was using a GTX 1070 (GP104 GPU) which has a Compute
Capability of 6.1). Additionally, make sure that the compiler path and executable
is correct. Once that is done, simply type 'make' and you should get an
executable named 'APSP'. To use the program, type './APSP graphfile', where
'graphfile' is a file that contains the necessary data, in the necessary format,
so as to represent the graph. Unlike its SSSP counterpart, output from the program
will only be on the screen. This is due to the fact that result file would be
prohibitively large on all but very small graphs. If you wish to add that feature,
though, a straightforward modification to 'template.cu' will accomplish that.

Sample graphs and the necessary instructions for creating the graph files can be
found in this file's parent directory.


Charles W. Johnson

April, 2018


