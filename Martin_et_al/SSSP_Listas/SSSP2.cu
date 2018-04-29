

/*******

The code below is the original code, edited so that it would run on CUDA
Compute Capability 6.1 hardware (EVGA/NVIDIA GTX 1070) with CUDA v9.0.176.
The display driver being used is NVIDIA 384.111. The OS is Debian Linux v9
('Sid').

Charles W Johnson
April, 2018

*******/


///////////////////////////////////////
///////////////////////////////// SSSP2
///////////////////////////////////////


/* CWJ includes */

#include <cuda.h>

#include "comun.cu"

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }



#include <unistd.h>

#ifndef _SSSP2
#define _SSSP2

bool ejecutarIteracion_SSSP2 ( 
                       const unsigned int nVuelta, 
                       const dim3 grid, const dim3 threads, 
                       const unsigned int nv, const unsigned int na,
                       const unsigned int mem_size_V, const unsigned int mem_size_A, 
                       const unsigned int mem_size_C, const unsigned int mem_size_F,
                       const unsigned int infinito,   
                       bool* p_h, bool* f_h, unsigned int* c_h ,
                       const unsigned int* v_d, const unsigned int* a_d, const unsigned int* w_d, 
                       bool* p_d, bool* f_d, unsigned int* c_d)
{
    //RECUERDA: mem_size_V= (nv+1)*sizeof(unsigned int)

    // ACTUALIZANDO CAMINOS MINIMOS ESPECIALES: kernel1

    cudaGetLastError(); // reset the runtime error variable to cudaSuccess

    kernel1<<<grid,threads>>>(v_d, a_d, w_d, p_d, f_d, c_d);

    // check if kernel execution generated and error
    gpuErrchk(cudaDeviceSynchronize());
    gpuErrchk(cudaPeekAtLastError());

    cudaThreadSynchronize();

    //traer C de device a host
    copiarD2H((void*)c_h, (void*)c_d, mem_size_C); 

    //MINIMIZANDO LOS COSTES RECIEN ACTUALIZADOS
    unsigned int i;
    unsigned int minimo = infinito;
    
    for (i=0; i<nv; i++) {
        if (p_h[i] && (minimo>c_h[i])) {
            minimo = c_h[i];
        }
    }

    //ACTUALIZANDO LA FRONTERA: Kernel3

    //ejecutar último kernel 
    cudaGetLastError(); // reset the runtime error variable to cudaSuccess

    kernel3<<<grid,threads>>>( p_d, f_d, c_d, minimo);

    // check if kernel execution generated and error
    gpuErrchk(cudaDeviceSynchronize());
    gpuErrchk(cudaPeekAtLastError());

    cudaThreadSynchronize();

    //Llevar de device a la actualizacion de P
    copiarD2H((void*) p_h, (void*)p_d, mem_size_F);
  
    return (minimo==infinito);

}//ejecutarIteracion_SSSP2


void testGraph_SSSP2(const unsigned int nv, const unsigned int mem_size_V,
                     const unsigned int na, const unsigned int mem_size_A,
                     const unsigned int infinito,   
                     const unsigned int* v_h, const unsigned int* a_h, const unsigned int* w_h,
                     const unsigned int* reference)
{
    //RECUERDA: mem_size_V= (nv+1)*sizeof(unsigned int)

    unsigned int* v_d; //array de vértices device
    unsigned int* a_d; //array de aristas device
    unsigned int* w_d; //array de pesos device
   
    //copiar grafo de host a device
    inicializar_Grafo_Device(v_h, mem_size_V, v_d, 
                             a_h, mem_size_A, a_d,
                             w_h, w_d);

    unsigned int* c_h; //solución en el host
    unsigned int* c_d; //solución en el device
    unsigned int mem_size_C= mem_size_V-sizeof(unsigned int); //Descontar el tapon -4
    inicializar_Sol(c_h, c_d, nv, mem_size_C, infinito);
    
    bool* f_h; //frontera en el host
    bool* f_d; //frontera en el device
    unsigned int mem_size_F= sizeof(bool) * nv;
    inicializar_Frontera(f_h, f_d, nv, mem_size_F);

    bool* p_h; //pendientes por procesar 
    bool* p_d; //pendientes por procesar 
    inicializar_Pendientes(p_h, p_d, nv, mem_size_F);

#ifdef DEBUG
    //DEPURACION
    printf("\nnv= %i\n", nv);
    printf("na= %i\n", na);
    printf("mem_size_V= %i\n", mem_size_V);
    printf("mem_size_A= %i\n", mem_size_A);
    printf("mem_size_F= %i\n\n", mem_size_F);
#endif // _DEBUG

    // setup execution parameters
    unsigned int num_threadsInBlock= NUM_THREADS_IN_BLOCK;
    //unsigned int num_blocksInGrid= nv/num_threadsInBlock;     // original code but the next line is better
    unsigned int num_blocksInGrid = (nv + (num_threadsInBlock-1)) / num_threadsInBlock;

    dim3  grid(num_blocksInGrid, 1, 1);
    dim3  threads(num_threadsInBlock, 1, 1);

    /* Updated timer code for CUDA 9 */

    cudaEvent_t timerStart, timerStop;
    float time;

    //EJECUTAR VUELTAS  
    bool ultima= false;
    unsigned int i= 0;

    // start things
    cudaEventCreate(&timerStart);
    cudaEventCreate(&timerStop);
    cudaEventRecord(timerStart, 0);

    while(!ultima){
        i++;
        ultima= ejecutarIteracion_SSSP2( i, 
                                         grid, threads, 
                                         nv, na,
                                         mem_size_V, mem_size_A, mem_size_C, mem_size_F,
                                         infinito, 
                                         p_h, f_h, c_h, 
                                         v_d, a_d, w_d, p_d, f_d, c_d);
    }//while

    cudaEventRecord(timerStop, 0);
    cudaEventSynchronize(timerStop);

    cudaEventElapsedTime(&time, timerStart, timerStop);
    cudaEventDestroy(timerStart);
    cudaEventDestroy(timerStop);
    //printf("%.6f", time);
    printf("Runtime for SSSP2 algorithm is: %.6f ms\n", time);

    copiarD2H((void*)c_h, (void*)c_d, mem_size_C); 

    // cleanup memory
    free(f_h);
    free(p_h);

    cudaFree(v_d);
    cudaFree(a_d);
    cudaFree(w_d);

    cudaFree(c_d);
    cudaFree(f_d);
    cudaFree(p_d);

    // check result
    //CUTBoolean res = cutComparei( (int*)reference, (int*)c_h, nv);
    //printf( "%s\t", (1 == res) ? "OK" : "FAILED");

    //mostrarUI(c_h, nv, "c_h");
    //mostrarUI(reference, nv, "reference");


    // cleanup memory
    free(c_h);

}

#endif //#ifndef _SSSP2


