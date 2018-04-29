

/*******

The code below is the original code, edited so that it would run on CUDA
Compute Capability 6.1 hardware (EVGA/NVIDIA GTX 1070) with CUDA v9.0.176.
The display driver being used is NVIDIA 384.111. The OS is Debian Linux v9
('Sid').

Charles W Johnson
April, 2018

*******/


///////////////////////////////////////
///////////////////////////////// SSSP5
/////////////////////// usando texturas
///////////////////////////////////////


/* CWJ includes */

#include <cuda.h>

#include "comun.cu"

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }

#ifndef _SSSP5_Texture_All
#define _SSSP5_Texture_All


//////////////////////////////////////////
bool ejecutarIteracion_SSSP5_tex_all( 
                       const unsigned int nVuelta, 
                       const dim3 grid, const dim3 threads,
                       const unsigned int nv, const unsigned int na,
                       const unsigned int mem_size_V, const unsigned int mem_size_A, 
                       const unsigned int mem_size_C, const unsigned int mem_size_F,
                       const unsigned int infinito, 
                       bool* p_h, bool* f_h, unsigned int* c_h ,
                       bool* p_d, bool* f_d, unsigned int* c_d,
                       unsigned int*  chi, unsigned int* cho, unsigned int*  cdi, unsigned int* cdo)
{
    //RECUERDA: mem_size_V= (nv+1)*sizeof(unsigned int)

#ifdef DEBUG    
    printf("\n\n*******************\n");    
    printf("\nVUELTA %i\n",nVuelta);
    mostrarUI(c_h, nv, "c_h");    
    mostrarB(f_h, nv, "f_h");    
    mostrarB(p_h, nv, "p_h");    

    printf("\nEJECUCION KERNEL 1\n");    
    printf("num_threadsInBlock= %i\n", threads.x);
    printf("num_blocksInGrid= %i\n", grid.x);
#endif // DEBUG

    //ejecutar último kernel 

    cudaGetLastError(); // reset the runtime error variable to cudaSuccess

    // ACTUALIZANDO CAMINOS MINIMOS ESPECIALES: kernel1
    kernel1_SSSP5_tex<<<grid,threads,threads.x*sizeof(unsigned int)>>>( p_d, f_d, c_d);

    // check if kernel execution generated and error
    gpuErrchk(cudaDeviceSynchronize());
    gpuErrchk(cudaPeekAtLastError());

    cudaThreadSynchronize();

#ifdef DEBUG  
    copiarD2H((void*)c_h, (void*)c_d, mem_size_C);
    mostrarUI(c_h, nv, "c_h");

    printf("\nEJECUCION KERNEL 2\n");    
#endif // DEBUG

    //MINIMIZANDO LOS COSTES RECIEN ACTUALIZADOS
    unsigned int min= infinito; 
    minimizar(nv, c_d, p_d, threads, infinito, chi, cho, cdi, cdo, min);

#ifdef DEBUG
    printf("\n\nEl minimo es %i\n", min);    

    printf("\nEJECUCION KERNEL 3\n");    
    printf("num_threadsInBlock= %i\n", threads.x);
    printf("num_blocksInGrid= %i\n", grid.x);
#endif // DEBUG

    //ejecutar último kernel 

    cudaGetLastError(); // reset the runtime error variable to cudaSuccess

    //ACTUALIZANDO LA FRONTERA: Kernel3
    kernel3_tex<<<grid,threads>>>( p_d, f_d, min);

    // check if kernel execution generated and error
    gpuErrchk(cudaDeviceSynchronize());
    gpuErrchk(cudaPeekAtLastError());

    cudaThreadSynchronize();

#ifdef DEBUG
    copiarD2H( (void*) p_h, (void*)p_d, mem_size_F);
    mostrarB(p_h, nv, "p_h");
    copiarD2H( (void*) f_h, (void*)f_d, mem_size_F);
    mostrarB(f_h, nv, "f_h");
#endif // DEBUG

    return (min==infinito);
}   

//////////////////////////////////
void testGraph_SSSP5_tex_all(const unsigned int nv, const unsigned int mem_size_V,
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

    //enlazar las texturas
    cudaBindTexture(0, textura_v, v_d, mem_size_V);
    cudaBindTexture(0, textura_a, a_d, mem_size_A);
    cudaBindTexture(0, textura_w, w_d, mem_size_A);

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

    //enlazar las texturas del algoritmo
    cudaBindTexture(0, textura_c, c_d, mem_size_C);
    //cudaBindTexture(0, textura_p, p_d, mem_size_F);
    //cudaBindTexture(0, textura_f, f_d, mem_size_F);

#ifdef DEBUG
    //DEPURACION
    printf("\nnv= %i\n", nv);
    printf("na= %i\n", na);
    printf("mem_size_V= %i\n", mem_size_V);
    printf("mem_size_A= %i\n", mem_size_A);
    printf("mem_size_F= %i\n\n", mem_size_F);
#endif // DEBUG

    // setup execution parameters
    unsigned int num_threadsInBlock= NUM_THREADS_IN_BLOCK;
    //unsigned int num_blocksInGrid= nv/num_threadsInBlock;     // original code, but the line below is better
    unsigned int num_blocksInGrid = (nv + (num_threadsInBlock-1)) / num_threadsInBlock;

    dim3  grid(num_blocksInGrid, 1, 1);
    dim3  threads(num_threadsInBlock, 1, 1);

    //RESERVAR ESPACIO PARA LA MINIMIZACION
    unsigned int nvi = nv/(2*num_threadsInBlock);
    unsigned int nvo = nvi/(2*num_threadsInBlock);
    unsigned int* cdi;
    unsigned int* cdo;
    cudaMalloc((void**) &cdi, nvi*sizeof(unsigned int));
    cudaMalloc((void**) &cdo, nvo*sizeof(unsigned int));
    unsigned int* chi = (unsigned int*) malloc(nvi*sizeof(unsigned int));
    unsigned int* cho = (unsigned int*) malloc(nvo*sizeof(unsigned int));

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
        ultima= ejecutarIteracion_SSSP5_tex_all( i, 
                          grid, threads, 
                          nv, na,
                          mem_size_V, mem_size_A, mem_size_C, mem_size_F,
                          infinito, 
                          p_h, f_h, c_h, 
                          p_d, f_d, c_d, 
                          chi, cho, cdi, cdo);
 

    }//while

    // end things
    cudaEventRecord(timerStop, 0);
    cudaEventSynchronize(timerStop);

    cudaEventElapsedTime(&time, timerStart, timerStop);
    cudaEventDestroy(timerStart);
    cudaEventDestroy(timerStop);
    printf("Runtime for SSSP5_Texture_All algorithm is: %.6f ms\n", time);

    copiarD2H((void*)c_h, (void*)c_d, mem_size_C); 

    //desenlazar las texturas
    cudaUnbindTexture(textura_v);
    cudaUnbindTexture(textura_a);
    cudaUnbindTexture(textura_w);

    // cleanup memory
    cudaFree(v_d);
    cudaFree(a_d);
    cudaFree(w_d);

    free(f_h);
    free(p_h);

    //desenlazar las texturas
    cudaUnbindTexture(textura_c);
    //cudaUnbindTexture(textura_p);
    //cudaUnbindTexture(textura_f);

    cudaFree(c_d);
    cudaFree(f_d);
    cudaFree(p_d);

    free(chi);
    free(cho);

    cudaFree(cdi);
    cudaFree(cdo);

    // check result
    //CUTBoolean res = cutComparei( (int*)reference, (int*)c_h, nv);
    //printf( "%s\t", (1 == res) ? "OK" : "FAILED");
    //mostrarUI(c_h, nv, "c_h");
    //mostrarUI(reference, nv, "reference");

    // cleanup memory
    free(c_h);
}


#endif //#ifndef _SSSP5_Texture_All


