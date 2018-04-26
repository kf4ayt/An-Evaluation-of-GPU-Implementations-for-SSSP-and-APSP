
In order to compile this program, edit the Makefile so that the GPU architecture
is correct (in my case, I was using a GTX 1070 (GP104 GPU) which has a Compute
Capability of 6.1). Additionally, make sure that the compiler path and executable
is correct. Once that is done, simply type 'make' and you should get an
executable named 'SSSP'. To use the program, type './SSSP graphfile', where
'graphfile' is a file that contains the necessary data, in the necessary format,
so as to represent the graph. Output from the program will be both on the screen
and in a file named 'result.txt'.

Sample graphs and the necessary instructions for creating the graph files can be
found in this file's parent directory.


Charles W. Johnson

April, 2018


