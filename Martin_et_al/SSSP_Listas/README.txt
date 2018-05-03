
In order to compile this program, edit the Makefile so that the GPU architecture
is correct (in my case, I was using a GTX 1070 (GP104 GPU) which has a Compute
Capability of 6.1). Additionally, make sure that the compiler path and executable
is correct.

BEFORE YOU TYPE 'make', you need to edit 'template.cu' so as to run the SSSP
implementation that you wish to use. Scroll down to the 'runTest_SSSP' function
and you will find the lines that execute the various implementations. Uncomment
the appropriate lines and comment out any others as necessary. Note: There are
many CPU/GPU implementations, plus some CPU-only, from which to choose. The file
'SSSP_descriptions.txt' gives some more detail. Please note that of the SSSP5/SSSP6
implementation 'suites', I only have the SSSP5 suite working - the documentation
says that the SSSP5 and SSSP6 are essentially the same, so for the tests, due to
the time crunch, when the SSSP6 suite did not fairly immediately work, I set it
aside and only used the SSSP5 suite. Thus, if you wish to use the SSSP6 suite,
you will have to do what is necessary, so as to fix it. Beyond that, there are
a number of other features that I did not explore that you may wish to.

Once you have edited 'template.cu' so as to select the SSSP implementations that
you wish to use, type 'make' and you should an executable named 'sssp-listas'.
To use the program, type './sssp-listas graphfile', where 'graphfile' is a file
that contains the necessary data, in the necessary format, so as to represent
the graph. Output from the program will be to the screen - each SSSP
implementation will print its runtime in milliseconds. Please note: the runtime
is the time spent actually executing the implementation - all of the time spent
allocating memory on the host and on the device, loading the graph into host
memory, copying it over to device memory, etc, is NOT included in the stated
runtime.

To create graphs for this set of executables, see the Python scripts in the
'Utilities' directory in the main directory of this respository. Alternately,
the code does include functions to create synthetic graphs. Utilizing those
functions is fairly straightforward, though it is up to the user to do that
which is necessary to utilize them.


Charles W. Johnson

April, 2018


