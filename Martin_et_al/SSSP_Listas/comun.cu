

/*******

The code below is the original code, edited so that it would run on CUDA
Compute Capability 6.1 hardware (EVGA/NVIDIA GTX 1070) with CUDA v9.0.176.
The display driver being used is NVIDIA 384.111. The OS is Debian Linux v9
('Sid').

Charles W Johnson
April, 2018

*******/


#ifndef _COMUN
#define _COMUN

#include <stdlib.h>
#include <stdio.h>
#include <string.h>

#include "tools.h"


/* CWJ includes */
#include <cuda.h>


/*
    Courtesy of user talonmies of StackOverflow

    For CUDA error checking
*/

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }

inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }

    //} else {
    //printf("success!");
    //}
}



///////////////////////////////////////////
void copiarH2D(void* v_d, const void* v_h, const unsigned int mem_size)
{
    cudaMemcpy(v_d, v_h, mem_size, cudaMemcpyHostToDevice);
}


void copiarD2H(void* v_h, const void* v_d, const unsigned int mem_size)
{ 
    cudaMemcpy(v_h, v_d, mem_size, cudaMemcpyDeviceToHost);
}

///////////////////////////////////
void inicializar_Grafo_Device(const unsigned int* v_h, const unsigned int mem_size_V, 
                              unsigned int*& v_d, 
                              const unsigned int* a_h, const unsigned int mem_size_A, 
                              unsigned int*& a_d,
                              const unsigned int* w_h, unsigned int* & w_d)
{
    
    cudaMalloc((void**) &v_d, mem_size_V);
    copiarH2D(v_d, v_h, mem_size_V);

    cudaMalloc((void**) &a_d, mem_size_A);
    copiarH2D(a_d, a_h, mem_size_A);

    cudaMalloc((void**) &w_d, mem_size_A);
    copiarH2D(w_d, w_h, mem_size_A);    
}

//////////////////////////////////////////////
void inicializar_Sol(unsigned int* & c_h, unsigned int* & c_d, 
                     const unsigned int nv, const unsigned int mem_size_V, 
                     const unsigned int infinito)
{
    // allocate mem for the result on host side
    c_h = (unsigned int*) malloc(mem_size_V);

    // initalize the memory
    c_h[0] = 0;

    for (unsigned int i = 1; i <= nv-1; i++) {
    	c_h[i]= infinito;
    }

    // allocate device memory for result
    cudaMalloc((void**) &c_d, mem_size_V);

    // copy host memory to device
    copiarH2D((void*) c_d, (void*) c_h, mem_size_V);

#ifdef _DEBUG    
    //mostrarUI(c_h, nv, "c_h");
#endif

}


///////////////////////////////////////////////7
void inicializar_Pendientes(bool* & p_h, bool* & p_d, 
                            const unsigned int nv, const unsigned int mem_size_F)
{
    // allocate host memory
    p_h = (bool*) malloc(mem_size_F);

    // initalize the memory
    p_h[0] = false;

    for (unsigned int i = 1; i <= nv-1; i++) {
    	p_h[i]= true;
    }

    // allocate device memory for procesar_d
    cudaMalloc((void**) &p_d, mem_size_F);

    // copy host memory to device
    copiarH2D((void*)p_d, (void*)p_h, mem_size_F);

#ifdef _DEBUG    
    //mostrarB(p_h, nv, "p_h");
#endif

}

//////////////////////////////////////////7
void inicializar_Frontera(bool* & f_h, bool* & f_d,
                          const unsigned int nv, const unsigned int mem_size_F)
{
    // allocate host memory  for f_h
    f_h = (bool*) malloc(mem_size_F);

    // initalize the memory
    f_h[0] = true;

    for (unsigned int i = 1; i <= nv-1; i++) {
    	f_h[i]= false;
    }

    // allocate device memory for f_d
    cudaMalloc((void**) &f_d, mem_size_F);

    // copy host memory to device
    copiarH2D((void*)f_d, (void*)f_h, mem_size_F);


#ifdef _DEBUG    
    //mostrarB(f_h, nv, "f_h");
#endif

}



#endif //#ifndef _COMUN
