

/*******

The code below is the original code, edited so that it would run on CUDA
Compute Capability 6.1 hardware (EVGA/NVIDIA GTX 1070) with CUDA v9.0.176.
The display driver being used is NVIDIA 384.111. The OS is Debian Linux v9
('Sid').

Charles W Johnson
April, 2018

*******/


/////////////////////////////////////////////////////
/////////////////////////////////////////////// SSSP8
///////////////////////////// (Modificación de SSSP3)
/////////////////////////////////////////////////////


/* CWJ includes */

#include <cuda.h>

#include "comun.cu"

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }

#ifndef _SSSP8
#define _SSSP8

bool ejecutarIteracion_SSSP8( 
                       const unsigned int nVuelta, 
                       const dim3 grid, const dim3 threads, const dim3 grid_minimizar,
                       const unsigned int nv, const unsigned int na,
                       const unsigned int mem_size_V, const unsigned int mem_size_A, 
                       const unsigned int mem_size_F, const unsigned int mem_size_Minimizar,
                       const unsigned int infinito, 
                       bool* p_h, bool* f_h, unsigned int* c_h ,
                       unsigned int*  minimoDelBloque_h,
                       const unsigned int* v_d, const unsigned int* a_d, const unsigned int* w_d, 
                       bool* p_d, bool* f_d, unsigned int* c_d,
                       unsigned int*  minimoDelBloque_d)
{
    //RECUERDA: mem_size_V= (nv+1)*sizeof(unsigned int)

#ifdef DEBUG    
    printf("\n\n*******************\n");    
    printf("\nVUELTA %i\n",nVuelta);
    mostrarUI(c_h, nv, "c_h");    
    mostrarB(f_h, nv, "f_h");    
    mostrarB(p_h, nv, "p_h");    
#endif // DEBUG

    //ejecutar último kernel 

    cudaGetLastError(); // reset the runtime error variable to cudaSuccess

    // ejecutar primer kernel
    kernel1_SSSP8<<<grid,threads>>>(v_d, a_d, w_d, p_d, f_d, c_d);

    // check if kernel execution generated and error
    gpuErrchk(cudaDeviceSynchronize());
    gpuErrchk(cudaPeekAtLastError());

    cudaThreadSynchronize();

#ifdef DEBUG  
    printf("\nEJECUCION KERNEL 1\n");    
    printf("num_threadsInBlock= %i\n", threads.x);
    printf("num_blocksInGrid= %i\n", grid.x);
    copiarD2H((void*)c_h, (void*)c_d, mem_size_V-sizeof(unsigned int)); //Descontar el tapon
    mostrarUI(c_h, nv, "c_h");
#endif // DEBUG

    //ejecutar último kernel 

    cudaGetLastError(); // reset the runtime error variable to cudaSuccess

    //minimizar
    kernel_minimizar1<<<grid_minimizar,threads,sizeof(int)*threads.x>>>(p_d, c_d, infinito,
                                                                        minimoDelBloque_d);

    // check if kernel execution generated and error
    gpuErrchk(cudaDeviceSynchronize());
    gpuErrchk(cudaPeekAtLastError());

    cudaThreadSynchronize();

    copiarD2H((void*)minimoDelBloque_h, (void*)minimoDelBloque_d, mem_size_Minimizar);
    unsigned int min = infinito;

#ifdef DEBUG
    printf("\nEJECUCION KERNEL 2\n");    
    printf("num_threadsInBlock= %i\n", threads.x);
    printf("num_blocksInGrid= %i\n", grid_minimizar.x);
    //mostrar minimoDelBloque_h y calcular su mínimo
    printf("\nminimoDelBloque_h[0..%i]\n", grid_minimizar.x-1);    

    for (unsigned int i = 0; i < grid_minimizar.x; i++) {
        printf("%i\t", minimoDelBloque_h[i]);

        if (min>minimoDelBloque_h[i]) {
            min = minimoDelBloque_h[i];
        }
    }

    printf("\n\nEl minimo es %i\n", min);    
#else // DEBUG
    for (unsigned int i = 0; i < grid_minimizar.x; i++) {
        if (min>minimoDelBloque_h[i]) {
            min = minimoDelBloque_h[i];
        }
    }
#endif // DEBUG

    //ejecutar último kernel 

    cudaGetLastError(); // reset the runtime error variable to cudaSuccess

    //ejecutar último kernel 
    kernel3<<<grid,threads>>>(p_d, f_d, c_d, min);

    // check if kernel execution generated and error
    gpuErrchk(cudaDeviceSynchronize());
    gpuErrchk(cudaPeekAtLastError());

    cudaThreadSynchronize();

#ifdef DEBUG
    printf("\nEJECUCION KERNEL 3\n");    
    printf("num_threadsInBlock= %i\n", threads.x);
    printf("num_blocksInGrid= %i\n", grid.x);
    copiarD2H((void*) p_h, (void*)p_d, mem_size_F);
    mostrarB(p_h, nv, "p_h");
    copiarD2H((void*) f_h, (void*)f_d, mem_size_F);
    mostrarB(f_h, nv, "f_h");
#endif // DEBUG

    return (min == infinito);
}   


/////////////////////////////////////// 
void testGraph_SSSP8(const unsigned int nv, const unsigned int mem_size_V,
                     const unsigned int na, const unsigned int mem_size_A,
                     const unsigned int infinito, const unsigned int* v_h,
                     const unsigned int* a_h, const unsigned int* w_h,
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
    unsigned int mem_size_C = mem_size_V-sizeof(unsigned int); //Descontar el tapon -4
    inicializar_Sol(c_h, c_d, nv, mem_size_C, infinito);
    
    bool* f_h; //frontera en el host
    bool* f_d; //frontera en el device
    unsigned int mem_size_F = sizeof(bool) * nv;
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
#endif // DEBUG

    // setup execution parameters
    unsigned int num_threadsInBlock = NUM_THREADS_IN_BLOCK;
    //unsigned int num_blocksInGrid= nv/num_threadsInBlock;     // original code, but the line below is better
    unsigned int num_blocksInGrid = (nv + (num_threadsInBlock-1)) / num_threadsInBlock;

    dim3  grid(num_blocksInGrid, 1, 1);
    dim3  threads(num_threadsInBlock, 1, 1);
    dim3  grid_minimizar((num_blocksInGrid/2), 1, 1);

    //RESERVAR ESPACIO PARA LA MINIMIZACIÓN DE LOS BLOQUES
    unsigned int mem_size_Minimizar = sizeof(unsigned int) * grid_minimizar.x;
    unsigned int* minimoDelBloque_h = (unsigned int*) malloc(mem_size_Minimizar);

    for (unsigned int i=0; i<grid_minimizar.x; i++) {
        minimoDelBloque_h[i] = infinito;
    }

    unsigned int* minimoDelBloque_d;
    cudaMalloc((void**) &minimoDelBloque_d, mem_size_Minimizar);
    copiarH2D((void*)minimoDelBloque_d, (void*)minimoDelBloque_h, mem_size_Minimizar);

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
        ultima= ejecutarIteracion_SSSP8( i, 
                          grid, threads, grid_minimizar,
                          nv, na,
                          mem_size_V, mem_size_A, mem_size_F, mem_size_Minimizar,
                          infinito,  
                                         p_h, f_h, c_h, minimoDelBloque_h,
                          v_d, a_d, w_d, p_d, f_d, c_d, minimoDelBloque_d);
    }//while

    // end things
    cudaEventRecord(timerStop, 0);
    cudaEventSynchronize(timerStop);

    cudaEventElapsedTime(&time, timerStart, timerStop);
    cudaEventDestroy(timerStart);
    cudaEventDestroy(timerStop);
    printf("Runtime for SSSP8 algorithm is: %.6f ms\n", time);

    copiarD2H((void*)c_h, (void*)c_d, mem_size_C);

    // cleanup memory
    free(f_h);
    free(p_h);
    free(minimoDelBloque_h);

    cudaFree(v_d);
    cudaFree(a_d);
    cudaFree(w_d);

    cudaFree(c_d);
    cudaFree(f_d);
    cudaFree(p_d);
    cudaFree(minimoDelBloque_d);

    // check result
    //CUTBoolean res = cutComparei( (int*)reference, (int*)c_h, nv);
    //printf( "%s\t", (1 == res) ? "OK" : "FAILED");
    //mostrarUI(c_h, nv, "c_h");
    //mostrarUI(reference, nv, "reference");

    // cleanup memory
    free(c_h);
}


#endif //#ifndef _SSSP8


