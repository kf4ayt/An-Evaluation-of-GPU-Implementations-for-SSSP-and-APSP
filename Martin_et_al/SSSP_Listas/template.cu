

/*******

The code below is the original code, edited so that it would run on CUDA
Compute Capability 6.1 hardware (EVGA/NVIDIA GTX 1070) with CUDA v9.0.176.
The display driver being used is NVIDIA 384.111. The OS is Debian Linux v9
('Sid').

Charles W Johnson
April, 2018

*******/


////////////
//CONSTANTES
////////////

#define NUM_THREADS_IN_BLOCK 256        //2^8 número de hilos por bloque
#define N_MEGAS 8                       //número de megas
#define N_NODOS N_MEGAS*1024*1024 

/*
    Note: N_NODOS must be a multiple of NUM_THREADS_IN_BLOCK so that grid blocks cover all vertices

    Note: N_NODOS must be a multiple of 2 * NUM_THREADS_IN_BLOCK for SSSP3

    Note: N_NODOS must be a multiple of 2 * NUM_THREADS_IN_BLOCK for SSSP4, SSSP5, and SSSP6
*/

#define DEGREE 7                //grado de libertad de los nodos
#define TOPE_W 10               //máximo peso de arista
#define INFINITO N_NODOS*TOPE_W //inicialización de la solución

#define NUM_GRAFOS 50


/////////////////////////////
//TEXTURAS PARA SSSP5 Y SSSP6
/////////////////////////////
//
// Textures to store graphs in
//
texture<unsigned int, 1, cudaReadModeElementType> textura_v; 
texture<unsigned int, 1, cudaReadModeElementType> textura_a; 
texture<unsigned int, 1, cudaReadModeElementType> textura_w; 

// Textures to store the structures that the algorithms require
//
texture<unsigned int, 1, cudaReadModeElementType> textura_c; 
texture<char1, 1, cudaReadModeElementType> textura_p; 
texture<char1, 1, cudaReadModeElementType> textura_f; 


//////////
//INCLUDES
//////////

// System Includes
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

// Project Includes

#include <math_functions.h>
//#include "tools.h"

#include "tools.cu"

// Includes for CPU-based algorithms

//#include "template_gold.h"
//#include "GeneradorGrafos.h"

#include "template_gold.cu"
#include "GeneradorGrafos.cu"

#include "Fibonacci_Heaps.cu"
#include "Fibonacci_Heaps.h"    // need for struct definitions

// File with kernels

#include "template_kernel.cu"


// General SSSP functions
#include "comun.cu"

// are working
#include "SSSP0.cu"
#include "SSSP1.cu"
#include "SSSP2.cu"
#include "SSSP3.cu"
#include "SSSP4.cu"
#include "SSSP5.cu"
#include "SSSP5_Texture.cu"
#include "SSSP5_Texture_All.cu"
#include "SSSP5_Texture_AllOfAll.cu"

// aren't working
#include "SSSP6.cu"
#include "SSSP6_Texture.cu"
#include "SSSP6_Texture_All.cu"
#include "SSSP6_Texture_AllOfAll.cu"

// are working
#include "SSSP8.cu"
#include "SSSP8_Atomic.cu"


/* CWJ includes */

#include <cuda.h>


///////////////////////
// Function declarations for functions below
///////////////////////
void runTest_SSSP(int argc, char** argv);
void runFH_SSSP();

//#include <new.h>
#include <malloc.h>

using namespace std;

////////////////////////////////////////////////////////////////////////////////
// Program main
////////////////////////////////////////////////////////////////////////////////

int main (int argc, char** argv) 
{
    //CUT_DEVICE_INIT(argc, argv);

    //test_Grafos(N_NODOS, DEGREE, TOPE_W);

    // generate a bunch of random graphs
    //generar_Grafos(N_MEGAS, NUM_GRAFOS, DEGREE, TOPE_W);

    //_set_new_mode(1);
	//unsigned int i= _HEAP_MAXREQ;
	//unsigned nTrozos= 4;
    //invertir_Grafos4( 14, 25, nTrozos, DEGREE);

    //imprimir_Degrees_Grafos( N_MEGAS, NUM_GRAFOS);


    runTest_SSSP(argc, argv);


    //runFH_SSSP();

    //CUT_EXIT(argc, argv);
    
    return 0;
}


///////////////////////////////////////////// 
void runTest_SSSP(int argc, char** argv)
{
    unsigned int degree= DEGREE;
    unsigned int topeW= TOPE_W;

    //lISTA DE PREDECESORES
    unsigned int* v1; //array de vértices host
    unsigned int nv1; //número de vértices 
    unsigned int mem_size_V1; //memoria del array con tapon

    unsigned int* a1; //array de aristas host
    unsigned int na1; //número de aristas
    unsigned int mem_size_A1; //memoria del array

    unsigned int* w1; //array de pesos host

    //lISTA DE SUCESORES
    unsigned int* v2; //array de vértices host
    unsigned int nv2; //número de vértices 
    unsigned int mem_size_V2; //memoria del array con tapon

    unsigned int* a2; //array de aristas host
    unsigned int na2; //número de aristas
    unsigned int mem_size_A2; //memoria del array

    unsigned int* w2; //array de pesos host


    // compute reference solution
    unsigned int* reference1;
    unsigned int* reference2;
    unsigned int* reference3;

    char s1[100];
    char s2[100];
    
    //DEPURACION
    printf("TEST SSSP\n\n");
    printf("threads per block = %i\n", NUM_THREADS_IN_BLOCK);
    //printf("hilos por bloque = %i\n", NUM_THREADS_IN_BLOCK);
    //printf("degree = %i\n", degree);
    //printf("topeW = %i\n\n", topeW);

    printf("\n");

    //RESULTADOS
    unsigned int num_grafos = NUM_GRAFOS;
    unsigned int n_megas = N_MEGAS;
    unsigned int infinito;
    //CUTBoolean res;
    
    //Fibonacci Heaps
    f_heap* fh = new f_heap;
    heap_node** hn;
    unsigned int x;

    for (unsigned int m = 14; m<=14; m++) {
        infinito= m*1024*1024*topeW; //actualizar inifinito para esta m
        //printf("\n\nNODOS= %i * 1024 * 1024\n\n", m);
        //printf("Grafo\t CPU3\t\t\t SSSP0\t\t\t SSSP1\t\t\t SSSP2\t\t\t SSSP3\t\t\t SSSP4\t\t\t SSSP5\t\t\t SSSP5_tex\t\t\t SSSP5_tex_all\t\t SSSP5_tex_allOfAll\t\t\t SSSP6\t\t\t SSSP6_tex\t\t\t SSSP6_tex_all\t\t SSSP6_tex_allOfAll\t\t\t CPU8\t\t\t SSSP8\t\t\t SSSP8_Atomic\n\n");
        //printf("Grafo\t CPU8\t\t\t FH\n\n");
      
      
        //Construcción de hn
        if ((hn = (heap_node**) malloc (m*1024*1024* sizeof(heap_node*))) == (heap_node**)NULL)
            exit(NOT_ENOUGH_MEM);	
		   
        for (x = 0; x<(m*1024*1024); x++) {
            hn[x] = new heap_node;
        }

        for (unsigned int i = 1; i<=1; i++) {

            //printf("%i\t", i);
            
            //if(i<10) sprintf(s1,"E:/CHUS/DATA/GRAFOS/%d/grafo0%d.gr", m, i);
            //else sprintf(s1,"E:/CHUS/DATA/GRAFOS/%d/grafo%d.gr", m, i);

            //if(i<10) sprintf(s2,"E:/CHUS/DATA/GRAFOS INVERTIDOS/%d/grafo0%d.gr", m, i);
            //else sprintf(s2,"E:/CHUS/DATA/GRAFOS INVERTIDOS/%d/grafo%d.gr", m, i);
            
            //if(i<10) sprintf(s2,"inverted_graphs/%d/grafo0%d.gr", m, i);
            //else sprintf(s2,"inverted_graphs/%d/grafo%d.gr", m, i);


            if (i<10) {
                //sprintf(s1,"./graphs/grafo-%d-0%d.gr", m, i);
            } else {
                sprintf(s1,"graphs/grafo-%d-%d.gr", m, i);
            }

            // lets me specify the graph to use via the cmd line if I so choose
            if (argc == 2) {
                sprintf(s1, argv[1]);
            }

            cout << "The graph file being used is: " << s1 << endl << endl;

            //Leer grafo
	    readGraphFromFile(s1, nv1, na1, mem_size_V1, mem_size_A1, v1, a1, w1);
	    nv1 = nv1-1; //descontar el tapon

            //Generar grafo
            //nv1= m*1024*1024;
            //generaGrafo(nv1, degree, topeW, mem_size_V1, na1, mem_size_A1, v1, a1, w1);

            // compute reference solution CPU3
            reference1 = (unsigned int*) malloc(mem_size_V1 - sizeof(unsigned int)); 
                                                            //Descontar el tapon

            //computeGold_SSSP3(reference1, nv1, v1, na1, a1, w1, infinito);


            //test del grafo


//            testGraph_SSSP0(nv1, mem_size_V1, na1, mem_size_A1, infinito, v1, a1, w1, reference1);

//            testGraph_SSSP1(nv1, mem_size_V1, na1, mem_size_A1, infinito, v1, a1, w1, reference1);

//            testGraph_SSSP2(nv1, mem_size_V1, na1, mem_size_A1, infinito, v1, a1, w1, reference1);

//            testGraph_SSSP3(nv1, mem_size_V1, na1, mem_size_A1, infinito, v1, a1, w1, reference1);

//            testGraph_SSSP4(nv1, mem_size_V1, na1, mem_size_A1, infinito, v1, a1, w1, reference1);

//            testGraph_SSSP5(nv1, mem_size_V1, na1, mem_size_A1, infinito, v1, a1, w1, reference1);

//            testGraph_SSSP5_tex(nv1, mem_size_V1, na1, mem_size_A1, infinito, v1, a1, w1, reference1);

//            testGraph_SSSP5_tex_all(nv1, mem_size_V1, na1, mem_size_A1, infinito, v1, a1, w1, reference1);

//            testGraph_SSSP5_tex_allOfAll(nv1, mem_size_V1, na1, mem_size_A1, infinito, v1, a1, w1, reference1);  

/**** I've decided that since SSSP6 functions mirror SSSP5 so closely,
      then I'm not going to spend time getting them to work unless I
      find that I have it.                                             ****/

            ////testGraph_SSSP6(nv1, mem_size_V1, na1, mem_size_A1, infinito, v1, a1, w1, reference1);
            ////testGraph_SSSP6_tex(nv1, mem_size_V1, na1, mem_size_A1, infinito, v1, a1, w1, reference1);
            ////testGraph_SSSP6_tex_all(nv1, mem_size_V1, na1, mem_size_A1, infinito, v1, a1, w1, reference1);
            ////testGraph_SSSP6_tex_allOfAll(nv1, mem_size_V1, na1, mem_size_A1, infinito, v1, a1, w1, reference1);  

            //Invertir grafo
            	//invertir_Grafo( nv1, na1, mem_size_V1, mem_size_A1, v1, a1, w1,
				//                nv2, na2, mem_size_V2, mem_size_A2, v2, a2, w2);


            // cleanup memory
            free(v1);
            free(a1);
            free(w1);


            //Leer grafo invertido
            //leeGrafo_FicheroB(s2, nv2, na2, mem_size_V2, mem_size_A2, v2, a2, w2);

            readGraphFromFile(s1, nv2, na2, mem_size_V2, mem_size_A2, v2, a2, w2);

            nv2 = nv2-1; //descontar el tapon

        
            // compute reference solution CPU8
            reference2 = (unsigned int*) malloc(mem_size_V2 - sizeof(unsigned int)); 
                                                                //Descontar el tapon                                      
            if (reference2 == (unsigned int*)NULL)
                exit(NOT_ENOUGH_MEM);
		    
            reference1 = (unsigned int*) malloc(mem_size_V2 - sizeof(unsigned int)); 
                                                                //Descontar el tapon
            if (reference1 == (unsigned int*)NULL)
                exit(NOT_ENOUGH_MEM);                                                            


            //computeGold_SSSP8(reference2, nv2, v2, na2, a2, w2, infinito);

            //computeGold_FH(reference1, nv2, v2, na2, a2, w2, infinito, fh, hn);


            // check result
            //res = cutComparei( (int*)reference2, (int*)reference1, nv2);
            //printf( "%s\t", (1 == res) ? "OK" : "FAILED");


            // cleanup memory
            free(reference1);


            //test del grafo
            testGraph_SSSP8(nv2, mem_size_V2, na2, mem_size_A2, infinito, v2, a2, w2, reference2);  

            testGraph_SSSP8_Atomic(nv2, mem_size_V2, na2, mem_size_A2, infinito, v2, a2, w2, reference2);  

printf("\n");

            // compute reference solution DIJKSTRA
            reference3 = (unsigned int*) malloc( mem_size_V2- sizeof(unsigned int)); 
                                                            //Descontar el tapon
            //computeGold_Dijkstra(reference3, nv2, v2, na2, a2, w2, infinito);


            // check result
            //res = cutComparei( (int*)reference2, (int*)reference3, nv2);
            //printf( "%s\t", (1 == res) ? "OK" : "FAILED");


            printf("\n");

            // cleanup memory
            free(v2);
            free(a2);
            free(w2);
            free(reference2);
            free(reference3);

        }//for i
        
        //destrucción del array hn
        for (x = 0; x<nv2; x++) {
            delete hn[x];
        }
        free(hn);
        hn = (heap_node**)NULL;
		
    }//for m
 
    delete fh;
}


// Hecho por Robertito para los grafos grandes
void runFH_SSSP(){

    unsigned int degree= DEGREE;
    unsigned int topeW= TOPE_W;

    //lISTA DE PREDECESORES
    unsigned int* v1; //array de vértices host
    unsigned int nv1; //número de vértices 
    unsigned int mem_size_V1; //memoria del array con tapon

    unsigned int* a1; //array de aristas host
    unsigned int na1; //número de aristas
    unsigned int mem_size_A1; //memoria del array

    unsigned int* w1; //array de pesos host

    //lISTA DE SUCESORES
    unsigned int* v2; //array de vértices host
    unsigned int nv2; //número de vértices 
    unsigned int mem_size_V2; //memoria del array con tapon

    unsigned int* a2; //array de aristas host
    unsigned int na2; //número de aristas
    unsigned int mem_size_A2; //memoria del array

    unsigned int* w2; //array de pesos host


    // compute reference solution
    unsigned int* reference1;
    unsigned int* reference2;
    //unsigned int* reference3;

    char s1[100];
    char s2[100];
    
    //DEPURACION
    printf("TEST SSSP\n\n");
    printf("hilos por bloque= %i\n", NUM_THREADS_IN_BLOCK);
    printf("degree= %i\n", degree);
    printf("topeW= %i\n\n", topeW);

    //RESULTADOS
    unsigned int num_grafos= NUM_GRAFOS;
    unsigned int n_megas= N_MEGAS;
    unsigned int infinito;
    //CUTBoolean res;
    
    //Fibonacci Heaps
    f_heap* fh= new f_heap;
    heap_node** hn;
    unsigned int x;

    for(unsigned int m = 14; m <= 14; m++){
        infinito= m*1024*1024*topeW; //actualizar inifinito para esta m
        printf("\n\nNODOS= %i * 1024 * 1024\n\n", m);
        //printf("Grafo\t CPU3\t\t\t SSSP0\t\t\t SSSP1\t\t\t SSSP2\t\t\t SSSP3\t\t\t SSSP4\t\t\t SSSP5\t\t\t SSSP5_tex\t\t\t SSSP5_tex_all\t\t SSSP5_tex_allOfAll\t\t\t SSSP6\t\t\t SSSP6_tex\t\t\t SSSP6_tex_all\t\t SSSP6_tex_allOfAll\t\t\t CPU8\t\t\t SSSP8\t\t\t SSSP8_Atomic\n\n");
        printf("Grafo\t CPU8\t\t\t FH\n\n");
      
		// Es cierto durante la primera vuelta del bucle
		bool primera_vez = true;
  
	    for(unsigned int i = 1; i <= 4; i++){

            printf("%i\t", i);
            //if(i<10) sprintf_s(s1,"graphs/%d/grafo0%d.gr", m, i);
            //else sprintf_s(s1,"graphs/%d/grafo%d.gr", m, i);

            //if(i<10) sprintf_s(s2,"inverted_graphs/%d/grafo0%d.gr", m, i);
            //else sprintf_s(s2,"inverted_graphs/%d/grafo%d.gr", m, i);
            
            if(i<10) sprintf(s2,"inverted_graphs/%d/grafo0%d.gr", m, i);
            else sprintf(s2,"inverted_graphs/%d/grafo%d.gr", m, i);

            //if(i<10) sprintf_s(s2,"./temp/%d/grafo0%d.gr", m, i);
            //else sprintf_s(s2,"./temp/%d/grafo%d.gr", m, i);

            //Leer grafo
	        //leeGrafo_FicheroB(s1, nv1, na1, mem_size_V1, mem_size_A1, v1, a1, w1);
	        //nv1= nv1-1; //descontar el tapon

            //Generar grafo
            //nv1= m*1024*1024;
            //generaGrafo( nv1, degree, topeW,  mem_size_V1, na1, mem_size_A1, v1, a1, w1);

            // compute reference solution CPU3
            //reference1= (unsigned int*) malloc( mem_size_V1- sizeof(unsigned int)); 
                                                            //Descontar el tapon
            //computeGold_SSSP3( reference1, nv1, v1, na1, a1, w1, infinito);


            //test del grafo
            //testGraph_SSSP0( nv1, mem_size_V1, na1, mem_size_A1, infinito, v1, a1, w1, reference1);

            //testGraph_SSSP1( nv1, mem_size_V1, na1, mem_size_A1, infinito, v1, a1, w1, reference1);

            //testGraph_SSSP2( nv1, mem_size_V1, na1, mem_size_A1, infinito, v1, a1, w1, reference1);

            //testGraph_SSSP3( nv1, mem_size_V1, na1, mem_size_A1, infinito, v1, a1, w1, reference1);

            //testGraph_SSSP4( nv1, mem_size_V1, na1, mem_size_A1, infinito, v1, a1, w1, reference1);

            //testGraph_SSSP5( nv1, mem_size_V1, na1, mem_size_A1, infinito, v1, a1, w1, reference1);

            //testGraph_SSSP5_tex( nv1, mem_size_V1, na1, mem_size_A1, infinito, v1, a1, w1, reference1);

            //testGraph_SSSP5_tex_all( nv1, mem_size_V1, na1, mem_size_A1, infinito, v1, a1, w1, reference1);

            //testGraph_SSSP5_tex_allOfAll( nv1, mem_size_V1, na1, mem_size_A1, infinito, v1, a1, w1, reference1);  

            //testGraph_SSSP6( nv1, mem_size_V1, na1, mem_size_A1, infinito, v1, a1, w1, reference1);

            //testGraph_SSSP6_tex( nv1, mem_size_V1, na1, mem_size_A1, infinito, v1, a1, w1, reference1);

            //testGraph_SSSP6_tex_all( nv1, mem_size_V1, na1, mem_size_A1, infinito, v1, a1, w1, reference1);

            //testGraph_SSSP6_tex_allOfAll( nv1, mem_size_V1, na1, mem_size_A1, infinito, v1, a1, w1, reference1);  


            //Invertir grafo
            //invertir_Grafo( nv1, na1, mem_size_V1, mem_size_A1, v1, a1, w1,
			//                nv2, na2, mem_size_V2, mem_size_A2, v2, a2, w2);


            // cleanup memory
            //free( v1);
            //free( a1);
            //free( w1);
/*
            //Leer grafo invertido
	        leeGrafo_FicheroB(s2, nv2, na2, mem_size_V2, mem_size_A2, v2, a2, w2);
	        nv2= nv2-1; //descontar el tapon
	        
            // compute reference solution CPU8
            reference2 = (unsigned int*) malloc( mem_size_V2- sizeof(unsigned int)); 
                                                            //Descontar el tapon                                      
		    if (reference2==(unsigned int*)NULL) exit ( NOT_ENOUGH_MEM );
		    
            reference1= (unsigned int*) malloc( mem_size_V2- sizeof(unsigned int)); 
                                                            //Descontar el tapon
		    if (reference1==(unsigned int*)NULL) exit ( NOT_ENOUGH_MEM );   
		    
		    computeGold_SSSP8( reference2, nv2, v2, na2, a2, w2, infinito);
		    
		    
			if( primera_vez ) {
			// if(true) {
				//Construcción de hn
				if ( (hn= (heap_node**) malloc ( m*1024*1024* sizeof(heap_node*) ) )
					==
					(heap_node**)NULL
				)
				exit ( NOT_ENOUGH_MEM );
			   
				for(x= 0; x<(m*1024*1024); x++)  hn[x]= new heap_node;
			}
			                                                        
                                                                                                                      
                                                            
            computeGold_FH( reference1, nv2, v2, na2, a2, w2, infinito, fh, hn);

            // check result
            res = cutComparei( (int*)reference2, (int*)reference1, nv2);
            printf( "%s\t", (1 == res) ? "OK" : "FAILED");
*/
            // cleanup memory
            free(reference1);

            //test del grafo
            //testGraph_SSSP8( nv2, mem_size_V2, na2, mem_size_A2, infinito, v2, a2, w2, reference2);  

            //testGraph_SSSP8_Atomic( nv2, mem_size_V2, na2, mem_size_A2, infinito, v2, a2, w2, reference2);  

            // compute reference solution DIJKSTRA
            //reference3 = (unsigned int*) malloc( mem_size_V2- sizeof(unsigned int)); 
                                                            //Descontar el tapon
            //computeGold_Dijkstra( reference3, nv2, v2, na2, a2, w2, infinito);

            // check result
            //res = cutComparei( (int*)reference2, (int*)reference3, nv2);
            //printf( "%s\t", (1 == res) ? "OK" : "FAILED");


            printf("\n");

            // cleanup memory
            free(v2);
            free(a2);
            free(w2);
            free(reference2);
            //free(reference3);

			/*
			//destrucción del array hn
			for(x= 0; x<(m*1024*1024); x++) delete hn[x];
			free(hn);
			hn= (heap_node**)NULL;
			*/

			// Al final del bucle deja de ser cierto primera_vez
			primera_vez = false;
        }//for i
        
        
		//destrucción del array hn
		for(x= 0; x<(m*1024*1024); x++) delete hn[x];
		free(hn);
		hn= (heap_node**)NULL;
		
		
		
    }//for m
    

	
	delete fh;

}


