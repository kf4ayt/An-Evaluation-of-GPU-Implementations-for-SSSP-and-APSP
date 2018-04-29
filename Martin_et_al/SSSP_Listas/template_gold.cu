

/*******

The code below is the original code, edited so that it would run on CUDA
Compute Capability 6.1 hardware (EVGA/NVIDIA GTX 1070) with CUDA v9.0.176.
The display driver being used is NVIDIA 384.111. The OS is Debian Linux v9
('Sid').

Charles W Johnson
April, 2018

*******/


/*
SOLUCIONES EN CPU
- computeGold_SSSP3: procesando los pendientes
- computeGold_SSSP8: procesando la frontera
*/

#include "template_gold.h"

#include <stdlib.h>
#include <stdio.h>
#include <string.h>

#include "tools.h"

/* CWJ includes */

#include <cuda.h>


/////////////////////////////////////////////////////////
/////////////////////////////////////////////////// SSSP3
/////////////////////////////////////////////////////////

bool ejecutarIteracion_SSSP3(const unsigned int nVuelta,
                             unsigned int* reference, 
                             unsigned int nv, const unsigned int* v, 
                             unsigned int na, const unsigned int* a, const unsigned int* w, 
                             const unsigned int infinito,
                             bool* f, bool* p)
{
    unsigned int i;

    //ACTUALIZACION DE REFERENCE USANDO LA FRONTERA
    unsigned int index; //indice inicial en A[...]
    unsigned int tope; //indice final en A[...]
    unsigned int pid; //nodo predecesor de i

    for (i=0; i<nv; i++) {
        if (p[i]) {
            //visitando los predecesores de i
            index = v[i];
            tope = v[i+1];

            while (index<tope) {
                pid = a[index];

                if (f[pid]) {
                    if (reference[i] > (reference[pid] + w[index])) {
                        reference[i] = reference[pid] + w[index];
                    }
                }

                index++;
            }
        }
    }


    //MINIMIZANDO LOS COSTES RECIEN ACTUALIZADOS
    unsigned int minimo = infinito;

    for (i=0; i<nv; i++) {
        if (p[i] && (minimo>reference[i])) {
            minimo = reference[i];
        }
    }

    //ACTUALIZANDO LA FRONTERA
    for (i=0; i<nv; i++) {
        f[i] = false;  

        if (p[i] && (reference[i]==minimo)) {
            f[i] = true;
            p[i] = false;
        }
    }

    return (minimo==infinito);

}//ejecutarIteracion


/////////////////////////////////////////////////////////////////
void computeGold_SSSP3(unsigned int* reference, 
                       const unsigned int nv, const unsigned int* v, 
                       const unsigned int na, const unsigned int* a, const unsigned int* w, 
                       const unsigned int infinito)
{
    //INICIALIZACION
    unsigned int mem_size_F = sizeof(bool) * nv;
    bool* p = (bool*) malloc(mem_size_F);
    bool* f = (bool*) malloc(mem_size_F);

    reference[0] = 0;
    p[0] = false;
    f[0] = true;

    for (unsigned int i=1; i<nv; i++) {
        reference[i] = infinito;
        p[i] = true;
        f[i] = false;
    }

    /* Updated timer code for CUDA 9 */

    cudaEvent_t timerStart, timerStop;
    float time;

    bool ultima = false;
    unsigned int i = 0;

    cudaEventCreate(&timerStart);
    cudaEventCreate(&timerStop);
    cudaEventRecord(timerStart, 0);

    while (!ultima)
    {
        i++;
        ultima = ejecutarIteracion_SSSP3(i, reference, nv, v, na, a, w, infinito, f, p);
    }

    cudaEventRecord(timerStop, 0);
    cudaEventSynchronize(timerStop);

    cudaEventElapsedTime(&time, timerStart, timerStop);
    cudaEventDestroy(timerStart);
    cudaEventDestroy(timerStop);
	
    printf("Runtime for computeGold_SSSP3 algorithm is: %.6f ms\n", time);

    //destrucción de arrays
    free(p);
    free(f);
}


/////////////////////////////////////////////////////////
/////////////////////////////////////////////////// SSSP8
/////////////////////////////////////////////////////////

bool ejecutarIteracion_SSSP8(const unsigned int nVuelta,
                             unsigned int* reference, 
                             unsigned int nv, const unsigned int* v, 
                             unsigned int na, const unsigned int* a, const unsigned int* w, 
                             const unsigned int infinito,
                             bool* f, bool* p)
{
    unsigned int tid;

    //ACTUALIZACION DE REFERENCE USANDO LA FRONTERA
    unsigned int i; //indice inicial en A[...]
    unsigned int tope; //indice final en A[...]
    unsigned int sid; //nodo predecesor de i

    for (tid=0; tid<nv; tid++) {
        if (f[tid]) { //tid está en la frontera
            //visitando los sucesores de tid
            i = v[tid];
            tope = v[tid+1];

            while (i<tope) {
                sid = a[i];

                if (p[sid]) { //sid está pendiente
                    if (reference[sid] > (reference[tid]+ w[i])) {
                        reference[sid] = reference[tid] + w[i];
                    }
                }

                i++;
            }
        }
    }

    //MINIMIZANDO LOS COSTES RECIEN ACTUALIZADOS
    unsigned int minimo = infinito;

    for (tid=0; tid<nv; tid++) {
        if (p[tid] && (minimo>reference[tid])) {
            minimo = reference[tid];
        }
    }

    //ACTUALIZANDO LA FRONTERA
    for (tid=0; tid<nv; tid++) {
        f[tid] = false;  

        if (p[tid] && (reference[tid]==minimo)) {
            f[tid] = true;
            p[tid] = false;
        }
    }

    return (minimo==infinito);

}//ejecutarIteracion


/////////////////////////////////////////////////////////
void computeGold_SSSP8(unsigned int* reference, 
                       const unsigned int nv, const unsigned int* v, 
                       const unsigned int na, const unsigned int* a, const unsigned int* w, 
                       const unsigned int infinito)
{
    //INICIALIZACION
    unsigned int mem_size_F = sizeof(bool) * nv;
    bool* p = (bool*) malloc(mem_size_F);
    bool* f = (bool*) malloc(mem_size_F);

    reference[0]= 0;
    p[0] = false;
    f[0] = true;

    for (unsigned int i= 1; i<nv; i++) {
        reference[i] = infinito;
        p[i] = true;
        f[i] = false;
    }

    /* Updated timer code for CUDA 9 */

    cudaEvent_t timerStart, timerStop;
    float time;

    bool ultima= false;
    unsigned int i= 0;

    cudaEventCreate(&timerStart);
    cudaEventCreate(&timerStop);
    cudaEventRecord(timerStart, 0);

    while (!ultima)
    {
        i++;
        ultima = ejecutarIteracion_SSSP8(i, reference, nv, v, na, a, w, infinito, f, p);
    }

    cudaEventRecord(timerStop, 0);
    cudaEventSynchronize(timerStop);

    cudaEventElapsedTime(&time, timerStart, timerStop);
    cudaEventDestroy(timerStart);
    cudaEventDestroy(timerStop);
	
    printf("Runtime for computeGold_SSSP8 algorithm is: %.6f ms\n", time);
	
    //destrucción de arrays
    free(p);
    free(f);
}

/////////////////////////////////////////////////////////
/////////////////////////////////////////////// DIJKSTRA
/////////////////////////////////////////////////////////

void computeGold_Dijkstra(unsigned int* reference, 
                          const unsigned int nv, const unsigned int* v, 
                          const unsigned int na, const unsigned int* a, const unsigned int* w, 
                          const unsigned int infinito)
{
    //INICIALIZACION
    unsigned int mem_size_F = sizeof(bool) * nv;
    bool* p = (bool*) malloc(mem_size_F);

    reference[0] = 0;
    p[0] = false;

    for (unsigned int i= 1; i<nv; i++) {
        reference[i] = infinito;
        p[i] = true;
    }

    unsigned int frontera = 0;

    /* Updated timer code for CUDA 9 */

    cudaEvent_t timerStart, timerStop;
    float time;

    unsigned int tid;
    bool ultima= false;
    unsigned int nVueltas= 0;

    cudaEventCreate(&timerStart);
    cudaEventCreate(&timerStop);
    cudaEventRecord(timerStart, 0);

    while(!ultima){
        nVueltas++;

        //ACTUALIZACION DE REFERENCE USANDO LA FRONTERA
        unsigned int i; //indice inicial en A[...]
        unsigned int tope; //indice final en A[...]
        unsigned int sid; //nodo predecesor de i
				
        i = v[frontera];
        tope = v[frontera+1];

        while (i<tope) {
            sid = a[i];

            if (p[sid]) { //sid está pendiente
                if (reference[sid] > (reference[frontera]+ w[i])) {
                    reference[sid] = reference[frontera] + w[i];
                }
            }

            i++;
        }

        //MINIMIZANDO LOS COSTES RECIEN ACTUALIZADOS
        unsigned int minimo = infinito;
        for (tid=0; tid<nv; tid++) {
            if (p[tid] && (minimo>reference[tid])) {
                minimo = reference[tid];
                frontera = tid;
            }
        }

        //ACTUALIZANDO LA FRONTERA
        if (minimo<infinito) {
            p[frontera] = false;
        } else {
            ultima = true;
        }
    }


    cudaEventRecord(timerStop, 0);
    cudaEventSynchronize(timerStop);

    cudaEventElapsedTime(&time, timerStart, timerStop);
    cudaEventDestroy(timerStart);
    cudaEventDestroy(timerStop);
	
    printf("Runtime for computeGold_Dijkstra algorithm is: %.6f ms\n", time);

    //destrucción de arrays
    free(p);
}


