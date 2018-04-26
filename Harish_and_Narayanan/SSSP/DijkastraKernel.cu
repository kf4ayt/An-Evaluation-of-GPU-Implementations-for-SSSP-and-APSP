
/*******

The code below is the original code, edited so that it would run on CUDA
Compute Capability 6.1 hardware (EVGA/NVIDIA GTX 1070) with CUDA v9.0.176.
The display driver being used is NVIDIA 384.111. The OS is Debian Linux v9
('Sid').

Charles W Johnson
April, 2018

*******/


/*************************************************************************************
Implementing Single Source Shortest Path on CUDA 1.1 Hardware using algorithm 
given in HiPC'07 paper "Accelerating Large Graph Algorithms on the GPU using CUDA"

Copyright (c) 2008 International Institute of Information Technology - Hyderabad. 
All rights reserved.
  
Permission to use, copy, modify and distribute this software and its documentation for 
educational purpose is hereby granted without fee, provided that the above copyright 
notice and this permission notice appear in all copies of this software and that you do 
not sell the software.
  
THE SOFTWARE IS PROVIDED "AS IS" AND WITHOUT WARRANTY OF ANY KIND,EXPRESS, IMPLIED OR 
OTHERWISE.

This Kernel updates the cost of each neighbour using atomicMin operation on CUDA 1.1 
hardware. Note that this operation is not supported on CUDA 1.0 hardware.

Created by Pawan Harish.
**************************************************************************************/

#include <stdio.h>

#include <cuda.h>
#include <cuda_runtime.h>

#define MAX_THREADS_PER_BLOCK 512

__global__ void DijkastraKernel1(int* g_graph_nodes, int* g_graph_edges, short int* g_graph_weights, 
				 int* g_graph_updating_cost, bool* g_graph_mask, 
				 int* g_cost, int no_of_nodes, int edge_list_size)
{
    int tid = blockIdx.x*MAX_THREADS_PER_BLOCK + threadIdx.x;
    int i, id;
    int end = edge_list_size;

    if ((tid < no_of_nodes) && g_graph_mask[tid])
    {
        if (tid < (no_of_nodes-1)) {
            end = g_graph_nodes[tid+1];
        }

	for (i = g_graph_nodes[tid]; i<end; i++)
        {
            id = g_graph_edges[i];
            atomicMin(&g_graph_updating_cost[id], g_cost[tid]+g_graph_weights[i]);
        }

        g_graph_mask[tid]=false;
    }
}


