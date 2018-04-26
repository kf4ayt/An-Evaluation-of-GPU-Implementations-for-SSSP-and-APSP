
/*******

The code below is the original code, edited so that it would run on CUDA
Compute Capability 6.1 hardware (EVGA/NVIDIA GTX 1070) with CUDA v9.0.176.
The display driver being used is NVIDIA 384.111. The OS is Debian Linux v9
('Sid').

Charles W Johnson
April, 2018

*******/


/***********************************************************************************
Implementing All Pairs Shortest Path on CUDA 1.1 Hardware using algorithm 
given in HiPC'07 paper "Accelerating Large Graph Algorithms on the GPU using CUDA"

Copyright (c) 2008 International Institute of Information Technology - Hyderabad. 
All rights reserved.
  
Permission to use, copy, modify and distribute this software and its documentation for 
educational purpose is hereby granted without fee, provided that the above copyright 
notice and this permission notice appear in all copies of this software and that you do 
not sell the software.
  
THE SOFTWARE IS PROVIDED "AS IS" AND WITHOUT WARRANTY OF ANY KIND,EXPRESS, IMPLIED OR 
OTHERWISE.

Created by Pawan Harish.
************************************************************************************/

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

#include <cuda.h>
#include <cuda_runtime.h>

#define MAX_THREADS_PER_BLOCK 512
#define MAX_COST 10000000

int no_of_nodes;
int edge_list_size;
FILE *fp;

#include "DijkastraKernel.cu"
#include "DijkastraKernel2.cu"

void DijGraph(int argc, char** argv);

////////////////////////////////////////////////////////////////////////////////
// Main Program
////////////////////////////////////////////////////////////////////////////////

int main(int argc, char** argv) 
{
    no_of_nodes=0;
    edge_list_size=0;

    DijGraph(argc, argv);

    return 0;
}



////////////////////////////////////////////////////////////////////////////////
//Apply Shortest Path on a Graph using CUDA
////////////////////////////////////////////////////////////////////////////////
void DijGraph( int argc, char** argv) 
{

    printf("Reading File\n");

    //Read in Graph from a file specified as a command line argument
    fp = fopen(argv[1], "r");

    if (!fp) {
        printf("Error Reading graph file\n");
        return;
    }

    int source = 0;

    fscanf(fp, "%d", &no_of_nodes);

    printf("No of Nodes: %d\n ", no_of_nodes);

    int num_of_blocks = 1;
    int num_of_threads_per_block = 1;

    //Make execution Parameters according to the number of nodes
    //Distribute threads across multiple Blocks if necessary

    if (no_of_nodes > MAX_THREADS_PER_BLOCK)
    {
        num_of_blocks = (no_of_nodes + (MAX_THREADS_PER_BLOCK-1)) / MAX_THREADS_PER_BLOCK;
        num_of_threads_per_block = MAX_THREADS_PER_BLOCK; 
    }

    // allocate host memory
    int* h_graph_nodes = (int*) malloc(sizeof(int)*no_of_nodes);
    bool *h_graph_mask = (bool*) malloc(sizeof(bool)*no_of_nodes);
    int *h_graph_updating_cost = (int*) malloc(sizeof(int)*no_of_nodes);

    int start, edgeno;   

    // initalize the memory
    int no=0;

    for (unsigned int i = 0; i < no_of_nodes; i++) 
    {
        fscanf(fp, "%d %d", &start, &edgeno);

        if (edgeno > 100) {
            no++;
        }

        h_graph_nodes[i] = start;
        h_graph_updating_cost[i] = MAX_COST;
        h_graph_mask[i]=false;
    }

    //read the source int from the file
    fscanf(fp, "%d", &source);

    //set the source int as true in the mask
    h_graph_mask[source]=true;

    fscanf(fp, "%d", &edge_list_size);
	
    printf("edgeListSize %d\n", edge_list_size);

    int id;
    int* h_graph_edges = (int*) malloc(sizeof(int)*edge_list_size);
    short int* h_graph_weights = (short int*) malloc(sizeof(short int)*edge_list_size);

    for (int i=0; i < edge_list_size; i++)
    {
        fscanf(fp,"%d",&id);
        h_graph_edges[i] = id;
        fscanf(fp,"%d",&id);
        h_graph_weights[i] = id;
    }

    if (fp) {
        fclose(fp);    
    }

    printf("Read File\n");

    printf("Total %d dense nodes, Avg Branching Factor: %f\n", no, edge_list_size/(float)no_of_nodes);

    //Copy the int list to device memory
    int* d_graph_nodes;
    cudaMalloc((void**) &d_graph_nodes, sizeof(int)*no_of_nodes);
    cudaMemcpy(d_graph_nodes, h_graph_nodes, sizeof(int)*no_of_nodes, cudaMemcpyHostToDevice);

    //Copy the Edge List to device Memory
    int* d_graph_edges;
    cudaMalloc((void**) &d_graph_edges, sizeof(int)*edge_list_size);
    cudaMemcpy(d_graph_edges, h_graph_edges, sizeof(int)*edge_list_size, cudaMemcpyHostToDevice);

    short int* d_graph_weights;
    cudaMalloc((void**) &d_graph_weights, sizeof(short int)*edge_list_size);
    cudaMemcpy(d_graph_weights, h_graph_weights, sizeof(short int)*edge_list_size, cudaMemcpyHostToDevice);

    //Copy the Mask to device memory
    bool* d_graph_mask;
    cudaMalloc((void**) &d_graph_mask, sizeof(bool)*no_of_nodes);
    cudaMemcpy(d_graph_mask, h_graph_mask, sizeof(bool)*no_of_nodes, cudaMemcpyHostToDevice);

    // allocate mem for the result on host side
    int* h_cost = (int*) malloc(sizeof(int)*no_of_nodes);

    for (int i=0; i < no_of_nodes; i++) {
        h_cost[i]= MAX_COST;
    }
    h_cost[source]=0;

    // allocate device memory for result
    int* d_cost;
    cudaMalloc((void**) &d_cost, sizeof(int)*no_of_nodes);
    cudaMemcpy(d_cost, h_cost, sizeof(int)*no_of_nodes, cudaMemcpyHostToDevice);

    int* d_graph_updating_cost;
    cudaMalloc((void**) &d_graph_updating_cost, sizeof(int)*no_of_nodes);
    cudaMemcpy(d_graph_updating_cost, h_graph_updating_cost, sizeof(int)*no_of_nodes, cudaMemcpyHostToDevice);

    //make a bool to check if the execution is over

    bool *d_finished;
    bool finished;
    cudaMalloc((void**) &d_finished, sizeof(bool));

    // setup execution parameters
    dim3  grid(num_of_blocks, 1, 1);
    dim3  threads(num_of_threads_per_block, 1, 1);

    // create the timer
    cudaEvent_t timerStart, timerStop;
    float time;

    cudaEventCreate(&timerStart);
    cudaEventCreate(&timerStop);

    // start the timer
    cudaEventRecord(timerStart, 0);

    int k=0;

    int* temp = (int*) malloc(sizeof(int)*no_of_nodes);

    for (int i=0; i < no_of_nodes; i++)
    {
        source = i;

        if (i>0)
        {
            h_cost[i-1]= MAX_COST;
            h_graph_mask[i-1]=false;
        }

        h_cost[source]=0;
        h_graph_mask[source]=true;

        cudaMemcpy(d_cost, h_cost, sizeof(int)*no_of_nodes, cudaMemcpyHostToDevice);
        cudaMemcpy(d_graph_mask, h_graph_mask, sizeof(bool)*no_of_nodes, cudaMemcpyHostToDevice);

        cudaMemcpy(d_graph_updating_cost, h_graph_updating_cost, sizeof(int)*no_of_nodes, cudaMemcpyHostToDevice);

        do
        {
            DijkastraKernel1<<<grid, threads>>>(d_graph_nodes, d_graph_edges, d_graph_weights, d_graph_updating_cost,
                                                d_graph_mask, d_cost, no_of_nodes, edge_list_size);

            k++;
            finished=false;

            cudaMemcpy(d_finished, &finished, sizeof(bool), cudaMemcpyHostToDevice);

            DijkastraKernel2<<<grid, threads>>>(d_graph_nodes, d_graph_edges, d_graph_weights, d_graph_updating_cost,
                                                d_graph_mask, d_cost, d_finished, no_of_nodes, edge_list_size);

            cudaMemcpy(&finished, d_finished, sizeof(bool), cudaMemcpyDeviceToHost);
        }
        while (finished);

        cudaMemcpy(temp, d_cost, sizeof(int)*no_of_nodes, cudaMemcpyDeviceToHost);
    }

    // stop the timer
    cudaEventRecord(timerStop, 0);
    cudaEventSynchronize(timerStop);

    cudaEventElapsedTime(&time, timerStart, timerStop);

    cudaEventDestroy(timerStart);
    cudaEventDestroy(timerStop);

    printf("Kernels Executed %d times\n",k);
    printf("Processing time: %f (ms)\n", time);

    // cleanup memory
    free(h_graph_nodes);
    free(h_graph_edges);
    free(h_graph_mask);
    free(h_graph_weights);
    free(h_graph_updating_cost);
    free(h_cost);
    free(temp);

    cudaFree(d_graph_nodes);
    cudaFree(d_graph_edges);
    cudaFree(d_graph_mask);
    cudaFree(d_graph_weights);
    cudaFree(d_graph_updating_cost);
    cudaFree(d_cost);
    cudaFree(d_finished);
}


