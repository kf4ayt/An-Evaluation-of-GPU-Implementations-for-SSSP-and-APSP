

/*******

The code below is the original code, edited so that it would run on CUDA
Compute Capability 6.1 hardware (EVGA/NVIDIA GTX 1070) with CUDA v9.0.176.
The display driver being used is NVIDIA 384.111. The OS is Debian Linux v9
('Sid').

Charles W Johnson
April, 2018

*******/


#include "GeneradorGrafos.h"


//generación de números aleatorios
#define _CRT_RAND_S

#include <stdlib.h>
#include <stdio.h>
#include <string>
#include <fstream>  //buscar otro fichero .h que haga lo corresponde


#include "tools.h"
#include "template_gold.h"


#include <limits.h>

#include "Lista.h"
#include "Arco.h"
#include "Nodo.h"

#include "Arco.cu"
#include "Nodo.cu"

// export C interface
//#include "GeneradorGrafos1.h"


/* CWJ includes */

#include "assert.h"
#include <cuda.h>
#include <iostream>
#include <time.h>
#include <climits>

#include <sstream>

#include <errno.h>

typedef int errno_t;


using namespace std;


//////////////////////IMPLEMENTATION

//MOSTRAR LISTA DE ADYACENCIA

unsigned int RangedRand(unsigned int range_min, unsigned int range_max)
{
   // Generate random numbers in the closed interval
   // [range_min, range_max]. In other words,
   // range_min <= random number <= range_max

    srand(time(NULL));

    unsigned int n = rand();

    if (n == 0) {
        printf("\n\nThe rand_s function failed!\n\n");
    }

    unsigned int u = (((double) n / (double) ULONG_MAX) * (range_max + 1 - range_min)) + range_min;
 
    while ((u<range_min) || (u>range_max)) {
        printf("\n\nFallo en la generación: %u\t%u\n\n", n, u);
        n = rand();
        if (n == 0) {
            printf("\n\nThe rand_s function failed!\n\n");
        }
        u = (((double) n / (double) UINT_MAX) * (range_max + 1 - range_min)) + range_min;
    }

    return u;
}


void generaGrafo(unsigned int nv, const unsigned int degree, 
                 const unsigned int topeW,  unsigned int&  mem_size_V,
                 unsigned int& na, unsigned int& mem_size_A,
                 unsigned int*& v, unsigned int*& a, unsigned int*& w)
{
    na = nv*degree;
    mem_size_A = na* sizeof(unsigned int);
    mem_size_V = (nv+1)* sizeof(unsigned int);

    v = (unsigned int*) malloc(mem_size_V);
    a = (unsigned int*) malloc(mem_size_A);
    w = (unsigned int*) malloc(mem_size_A);

    unsigned int nodo;
    unsigned int d;
    unsigned int k;
    unsigned int i;
    bool enc;

    for (i=0; i<nv; i++) {
        v[i]= i*degree;
        d=0;

        while (d < (int)degree) {
            nodo = RangedRand(0,nv-1);

            if (d == 0) {
                enc = false;
            } else {        // d>0
                k = 0;
                enc = (a[i*degree + k] == nodo);

                while ((k<(d-1)) && (!enc)) {
                    k++;
                    enc = (a[i*degree + k] == nodo);
                }
            }

            if (!enc) {
                a[i*degree + d] = nodo;
                w[i*degree + d] = RangedRand(1,topeW);
                d++;
            }

            d++;
        }
    }

    //incluir tapón
    v[nv] = na;


#ifdef DEBUG
    printf("nv = %i\n", nv);
    printf("na = %i\n", na);
    printf("mem_size_V = %i\n", mem_size_V);
    printf("mem_size_A = %i\n", mem_size_A);
    mostrarUI(v, nv+1, "v");
    mostrarUI(a, na, "a");
#endif //DEBUG


}


//Formato de fichero BINARIO
void guardaGrafo_FicheroB(const char* filename, const unsigned int nv, const unsigned int na,
                          const unsigned int mem_size_V, const unsigned int mem_size_A,
                          const unsigned int* v, const unsigned int* a, const unsigned int* w)
{
    ofstream outfile;
    outfile.open(filename, ios::binary);
    assert(outfile);

    outfile.write((char*)& nv, sizeof(unsigned int));
    outfile.write((char*)& na, sizeof(unsigned int));
    outfile.write((char*)v, mem_size_V);
    outfile.write((char*)a, mem_size_A);
    outfile.write((char*)w, mem_size_A);
    outfile.close();
}


// NOT binary
void writeGraphToFile(const char* filename, const unsigned int nv, const unsigned int na,
                      const unsigned int mem_size_V, const unsigned int mem_size_A,
                      const unsigned int* v, const unsigned int* a, const unsigned int* w)
{
    FILE* outfile;
    outfile = fopen(filename, "w");
    assert(outfile);

    fprintf(outfile, "%d %d\n", nv, na);   

    fprintf(outfile, "\n");   

    for(int i=0; i<nv; i++) {
        fprintf(outfile, "%d\n", v[i]);
    }

    fprintf(outfile, "\n");   

    for(int i=0; i<na; i++) {
        fprintf(outfile, "%d\n", a[i]);
    }

    fprintf(outfile, "\n");   

    for(int i=0; i<na; i++) {
        fprintf(outfile, "%d\n", w[i]);
    }

    fclose(outfile);
}


// NOT binary
void readGraphFromFile(const char* filename, 
                       unsigned int& nv, unsigned int& na, 
                       unsigned int& mem_size_V, unsigned int& mem_size_A,
                       unsigned int*& v, unsigned int*& a, unsigned int*& w)
{
    FILE* infile;
    infile = fopen(filename, "r");
    assert(infile);

    // get nv and na
    fscanf(infile, "%d %d", &nv, &na);

    mem_size_V = nv* sizeof(unsigned int);
    mem_size_A = na* sizeof(unsigned int);

    v = (unsigned int*) malloc(mem_size_V);
    a = (unsigned int*) malloc(mem_size_A);
    w = (unsigned int*) malloc(mem_size_A);	

    // perror("ERROR MEMORIA");
    if (v == (unsigned int*)NULL)
        exit(NOT_ENOUGH_MEM);
    if (a == (unsigned int*)NULL)
        exit(NOT_ENOUGH_MEM);
    if (w == (unsigned int*)NULL)
        exit(NOT_ENOUGH_MEM);

    for (int i=0; i<nv; i++) {
        fscanf(infile, "%d", &v[i]);
    }

    for (int i=0; i<na; i++) {
        fscanf(infile, "%d", &a[i]);
    }

    for (int i=0; i<na; i++) {
        fscanf(infile, "%d", &w[i]);
    }

    fclose(infile);
}


//Formato de fichero BINARIO específico para evitar la construcción de la matriz de adyacencia
void leeGrafo_FicheroB(const char* filename, 
                       unsigned int& nv, unsigned int& na, 
                       unsigned int& mem_size_V, unsigned int& mem_size_A,
                       unsigned int*& v, unsigned int*& a, unsigned int*& w)
{
    v = NULL;
    a = NULL;
    w = NULL;

    ifstream infile;
    infile.open(filename, ios::binary);
    assert(infile);

    infile.read((char*)& nv, sizeof(unsigned int));
    infile.read((char*)& na, sizeof(unsigned int));
    mem_size_V = nv* sizeof(unsigned int);
    mem_size_A = na* sizeof(unsigned int);

    v = (unsigned int*) malloc(mem_size_V);
    a = (unsigned int*) malloc(mem_size_A);
    w = (unsigned int*) malloc(mem_size_A);	

    // perror("ERROR MEMORIA");
    if (v == (unsigned int*)NULL)
        exit(NOT_ENOUGH_MEM);
    if (a == (unsigned int*)NULL)
        exit(NOT_ENOUGH_MEM);
    if (w == (unsigned int*)NULL)
        exit(NOT_ENOUGH_MEM);

    infile.read((char*)v, mem_size_V);
    infile.read((char*)a, mem_size_A);
    infile.read((char*)w, mem_size_A);
    infile.close();
}


////////////////////////////////////////////////////////////
//INVERSION DE GRAFOS REPRESENTADOS POR LISTAS DE ADYACENCIA
////////////////////////////////////////////////////////////

//Invertir un grafo
void invertir_Grafo(const unsigned int nv1, const unsigned int na1, 
					const unsigned int mem_size_V1, const unsigned int mem_size_A1,
					const unsigned int* v1, const unsigned int* a1, const unsigned int* w1,
					unsigned int& nv2, unsigned int& na2, 
					unsigned int& mem_size_V2, unsigned int& mem_size_A2,
					unsigned int*& v2, unsigned int*& a2, unsigned int*& w2)
{
    //RECUERDA: mem_size_V1= (nv1+1)*sizeof(int)

	nv2= nv1;
	na2= na1;
	mem_size_V2= mem_size_V1;
	mem_size_A2= mem_size_A1;

	v2= (unsigned int*) malloc(mem_size_V2);
	a2= (unsigned int*) malloc(mem_size_A2);
	w2= (unsigned int*) malloc(mem_size_A2);

	for(unsigned int i= 0; i<nv2; i++){
		v2[i]= 0;
	}//for

	unsigned int origen;
	for(unsigned int destino= 0; destino<nv1; destino++){
		for(unsigned int d= v1[destino]; d<v1[destino+1]; d++){
			//arista destino <- a1[d]
			origen= a1[d];
			v2[origen]= v2[origen]+1;
		}//for d
	}//for destino


	unsigned int* index= (unsigned int*) malloc(nv2* sizeof(unsigned int));

	unsigned int aux;
	unsigned int cont= 0;
	for(unsigned int i= 0; i<nv2; i++){
		aux= v2[i];
		v2[i]= cont;
		cont= cont + aux;
		index[i]= v2[i];
	}//for

	//incluir tapón
	v2[nv2]= na2;

	unsigned int peso;
	for(unsigned int destino= 0; destino<nv1; destino++){
		for(unsigned int d= v1[destino]; d<v1[destino+1]; d++){
			//arista destino <- a1[d]
			//peso w1[d]
			origen= a1[d];
			peso= w1[d];

			a2[index[origen]]= destino;
			w2[index[origen]]= peso;

			index[origen]= index[origen]+1;
		}//for d
	}//for o


	free(index);



#ifdef _DEBUG
    printf("nv1 = %i\n", nv1);
    printf("na1 = %i\n", na1);
    printf("mem_size_V1 = %i\n", mem_size_V1);
    printf("mem_size_A1 = %i\n", mem_size_A1);
    mostrarUI(v1, nv1+1, "v1");
    mostrarUI(a1, na1, "a1");

    printf("\n\n\n");
    printf("nv2 = %i\n", nv2);
    printf("na2 = %i\n", na2);
    printf("mem_size_V2 = %i\n", mem_size_V2);
    printf("mem_size_A2 = %i\n", mem_size_A2);
    mostrarUI(v2, nv2+1, "v2");
    mostrarUI(a2, na2, "a2");
#endif //_DEBUG

}


//Invertir todos los grafos
void invertir_Grafos(const unsigned int n_Megas, const unsigned int n_Grafos)
{
    //lISTA DE PRECDECESORES
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

    char s1[100];
    char s2[100];
    
    //DEPURACION
    printf("Invirtiendo los grafos\n\n");

    for (unsigned int m=12; m<=n_Megas; m++)
    {
        printf("\n\nMegas= %d\n\n", m);
        printf("Grafo\t Lectura\t Inversion\t Escritura\n\n");

        for (unsigned int i=2; i<=n_Grafos; i++)
        {
            printf("%i\t", i);

            if(i<10) sprintf(s1,"graphs/%d/grafo0%d.gr", m, i);
            else sprintf(s1,"graphs/%d/grafo%d.gr", m, i);

            if(i<10) sprintf(s2,"inverted_graphs/%d/grafo0%d.gr", m, i);
            else sprintf(s2,"inverted_graphs/%d/grafo%d.gr", m, i);

            //if(i<10) sprintf_s(s1,"E:/Chus/DATA/GRAFOS/%d/grafo0%d.gr", m, i);
            //else sprintf_s(s1,"E:/Chus/DATA/GRAFOS/%d/grafo%d.gr", m, i);

            //if(i<10) sprintf_s(s2,"E:/Chus/DATA/GRAFOS INVERTIDOS/%d/grafo0%d.gr", m, i);
            //else sprintf_s(s2,"E:/Chus/DATA/GRAFOS INVERTIDOS/%d/grafo%d.gr", m, i);

            /* Updated timer code for CUDA 9 */

            cudaEvent_t timerStart, timerStop;
            float time;

            // start things
            cudaEventCreate(&timerStart);
            cudaEventCreate(&timerStop);
            cudaEventRecord(timerStart, 0);

            //Leer grafo
            readGraphFromFile(s1, nv1, na1, mem_size_V1, mem_size_A1, v1, a1, w1);
            nv1 = nv1-1; //descontar el tapon

            // end things
            cudaEventRecord(timerStop, 0);
            cudaEventElapsedTime(&time, timerStart, timerStop);
            cudaEventDestroy(timerStart);
            cudaEventDestroy(timerStop);
            printf("%.6f", time);


            // start things
            cudaEventCreate(&timerStart);
            cudaEventCreate(&timerStop);
            cudaEventRecord(timerStart, 0);

            //Invertir grafo
            invertir_Grafo(nv1, na1, mem_size_V1, mem_size_A1, v1, a1, w1,
                           nv2, na2, mem_size_V2, mem_size_A2, v2, a2, w2);

            // end things
            cudaEventRecord(timerStop, 0);
            cudaEventElapsedTime(&time, timerStart, timerStop);
            cudaEventDestroy(timerStart);
            cudaEventDestroy(timerStop);
            printf("%.6f", time);


            // start things
            cudaEventCreate(&timerStart);
            cudaEventCreate(&timerStop);
            cudaEventRecord(timerStart, 0);
	
            //guardaGrafo_FicheroB(s2, nv2+1, na2, mem_size_V2, mem_size_A2, v2, a2, w2);
            writeGraphToFile(s2, nv2+1, na2, mem_size_V2, mem_size_A2, v2, a2, w2);

            // end things
            cudaEventRecord(timerStop, 0);
            cudaEventElapsedTime(&time, timerStart, timerStop);
            cudaEventDestroy(timerStart);
            cudaEventDestroy(timerStop);
            printf("%.6f", time);

            printf("%\n");

            // cleanup memory
            free(v1);
            free(a1);
            free(w1);

            // cleanup memory
            free(v2);
            free(a2);
            free(w2);
        }
    }
}

//Invertir un grafo usando arrays de listas, en lugar de un único array,
//para no precisar memoria RAM contigua
void invertir_Grafo2(const unsigned int nv1, const unsigned int na1, 
                     const unsigned int mem_size_V1, const unsigned int mem_size_A1,
                     unsigned int* v1, unsigned int* a1, unsigned int* w1,
                     unsigned int& nv2, unsigned int& na2, 
                     unsigned int& mem_size_V2, unsigned int& mem_size_A2,
                     unsigned int*& v2, unsigned int*& a2, unsigned int*& w2)
{
    Lista<Arco>** grafo_invertido= new Lista<Arco>* [nv1];

    unsigned int l;

    for (l=0; l<nv1; l++) {
        grafo_invertido[l] = new Lista<Arco>();
    }
	
	unsigned int origen;
	unsigned int destino;
	unsigned int peso;
	Arco* a;
	for(destino= 0; destino<nv1; destino++){
		for(unsigned int d= v1[destino]; d<v1[destino+1]; d++){
			//arista destino <- a1[d]
			origen= a1[d];
			peso= w1[d];
			a= new Arco(destino,peso);
			grafo_invertido[origen]->insertaFinal(a);
		}//for d
	}//for destino

	
	// cleanup memory, antes de construir v2, a2 y w2
	free( v1);
	free( a1);
	free( w1);

	//RECUERDA: mem_size_V1= (nv1+1)*sizeof(int)
	nv2 = nv1;
	na2 = na1;
	mem_size_V2 = mem_size_V1;
	mem_size_A2 = mem_size_A1;

	v2 = (unsigned int*) malloc(mem_size_V2);
	a2 = (unsigned int*) malloc(mem_size_A2);
	w2 = (unsigned int*) malloc(mem_size_A2);

	//incluir tapón
	v2[nv2] = na2;

	unsigned int cont= 0;
	Lista<Arco>* lista;
	for(origen= 0; origen<nv2; origen++){
		v2[origen]= cont;
		lista= grafo_invertido[origen];
		lista->inicia();
		while(!lista->final()){
			a= lista->getActual();
			lista->avanza();

			destino= a->getDestino();
			peso= a->getPeso();

			a2[cont]= destino;
			w2[cont]= peso;

			cont++;
		}//while

		//Aprovechamos para ir limpiando
		delete lista;
	}//for o

/*
#ifdef _DEBUG
    printf("\n\n\n");
    printf("nv2 = %i\n", nv2);
    printf("na2 = %i\n", na2);
    printf("mem_size_V2 = %i\n", mem_size_V2);
    printf("mem_size_A2 = %i\n", mem_size_A2);
    mostrarUI(v2, nv2+1, "v2");
    mostrarUI(a2, na2, "a2");
#endif //_DEBUG
*/

    // cleanup memory
    //for (l=0; l<nv1; l++) {
    //    delete grafo_invertido[l];
    //}

    delete[] grafo_invertido;
}


//Invertir todos los grafos: v1, a1 y w1 se eliminan antes de construir v2, a2 y w2
void invertir_Grafos2(const unsigned int n_Megas, const unsigned int n_Grafos)
{
	//lISTA DE PRECDECESORES
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

    char s1[100];
    char s2[100];
    
    //DEPURACION
    printf("Invirtiendo los grafos\n\n");


    for (unsigned int m=12; m<=n_Megas; m++)
    {
        printf("\n\nMegas= %d\n\n", m);
        printf("Grafo\t Lectura\t Inversion\t Escritura\n\n");

        for (unsigned int i=2; i<=n_Grafos; i++)
        {
            printf("%i\t", i);

            if(i<10) sprintf(s1,"graphs/%d/grafo0%d.gr", m, i);
            else sprintf(s1,"graphs/%d/grafo%d.gr", m, i);

            if(i<10) sprintf(s2,"inverted_graphs/%d/grafo0%d.gr", m, i);
            else sprintf(s2,"inverted_graphs/%d/grafo%d.gr", m, i);

            //if(i<10) sprintf_s(s1,"E:/Chus/DATA/GRAFOS/%d/grafo0%d.gr", m, i);
            //else sprintf_s(s1,"E:/Chus/DATA/GRAFOS/%d/grafo%d.gr", m, i);

            //if(i<10) sprintf_s(s2,"E:/Chus/DATA/GRAFOS INVERTIDOS/%d/grafo0%d.gr", m, i);
            //else sprintf_s(s2,"E:/Chus/DATA/GRAFOS INVERTIDOS/%d/grafo%d.gr", m, i);

            /* Updated timer code for CUDA 9 */

            cudaEvent_t timerStart, timerStop;
            float time;

            // start things
            cudaEventCreate(&timerStart);
            cudaEventCreate(&timerStop);
            cudaEventRecord(timerStart, 0);

            //Leer grafo
            readGraphFromFile(s1, nv1, na1, mem_size_V1, mem_size_A1, v1, a1, w1);
            nv1 = nv1-1; //descontar el tapon

            // end things
            cudaEventRecord(timerStop, 0);
            cudaEventElapsedTime(&time, timerStart, timerStop);
            cudaEventDestroy(timerStart);
            cudaEventDestroy(timerStop);
            printf("%.6f", time);


            // start things
            cudaEventCreate(&timerStart);
            cudaEventCreate(&timerStop);
            cudaEventRecord(timerStart, 0);

/*
#ifdef _DEBUG
    printf("nv1 = %i\n", nv1);
    printf("na1 = %i\n", na1);
    printf("mem_size_V1 = %i\n", mem_size_V1);
    printf("mem_size_A1 = %i\n", mem_size_A1);
    mostrarUI(v1, nv1+1, "v1");
    mostrarUI(a1, na1, "a1");
#endif //_DEBUG
*/

            //Invertir grafo: OJO v1, a1 y w1 se destruyen!
            invertir_Grafo2(nv1, na1, mem_size_V1, mem_size_A1, v1, a1, w1,
                            nv2, na2, mem_size_V2, mem_size_A2, v2, a2, w2);


            // end things
            cudaEventRecord(timerStop, 0);
            cudaEventElapsedTime(&time, timerStart, timerStop);
            cudaEventDestroy(timerStart);
            cudaEventDestroy(timerStop);
            printf("%.6f", time);


            // start things
            cudaEventCreate(&timerStart);
            cudaEventCreate(&timerStop);
            cudaEventRecord(timerStart, 0);
	
            writeGraphToFile(s2, nv2+1, na2, mem_size_V2, mem_size_A2, v2, a2, w2);


            // end things
            cudaEventRecord(timerStop, 0);
            cudaEventElapsedTime(&time, timerStart, timerStop);
            cudaEventDestroy(timerStart);
            cudaEventDestroy(timerStop);
            printf("%.6f", time);


            printf( "%\n");

            // cleanup memory
            free(v2);
            free(a2);
            free(w2);
        }
    }
}


//Invertir un grafo usando la clase Nodo para ahorrar memoria
void invertir_Grafo3(const unsigned int nv1, const unsigned int na1, 
                     const unsigned int mem_size_V1, const unsigned int mem_size_A1,
                     unsigned int* v1, unsigned int* a1, unsigned int* w1,
                     unsigned int& nv2, unsigned int& na2, 
                     unsigned int& mem_size_V2, unsigned int& mem_size_A2,
                     unsigned int*& v2, unsigned int*& a2, unsigned int*& w2)
{
    Nodo** grafo_invertido= new Nodo* [nv1];

    unsigned int l;

    for (l=0; l<nv1; l++) {
        grafo_invertido[l] = NULL;
    }
	
	unsigned int origen;
	unsigned int destino;
	unsigned int peso;
	Nodo* aux;
	for(destino= 0; destino<nv1/2; destino++){
		for(unsigned int d= v1[destino]; d<v1[destino+1]; d++){
			//arista destino <- a1[d]
			origen= a1[d];
			peso= w1[d];
			aux= grafo_invertido[origen];
			grafo_invertido[origen]= new Nodo(destino,peso,aux);
		}//for d
	}//for destino

	
	// cleanup memory, antes de construir v2, a2 y w2
	free( v1);
	free( a1);
	free( w1);

	//RECUERDA: mem_size_V1= (nv1+1)*sizeof(int)
	nv2= nv1;
	na2= na1;
	mem_size_V2= mem_size_V1;
	mem_size_A2= mem_size_A1;

	v2= (unsigned int*) malloc(mem_size_V2);
	a2= (unsigned int*) malloc(mem_size_A2);
	w2= (unsigned int*) malloc(mem_size_A2);

	//incluir tapón
	v2[nv2]= na2;

	unsigned int cont= 0;
	Nodo* lista;
	for(origen= 0; origen<nv2; origen++){
		v2[origen]= cont;
		lista= grafo_invertido[origen];
		while(lista!=NULL){
			destino= lista->getDestino();
			peso= lista->getPeso();

			a2[cont]= destino;
			w2[cont]= peso;

			aux= lista->getSig();
			delete lista; //vamos lilberando memoria
			lista= aux;

			cont++;
		}//while

	}//for o

/*
#ifdef _DEBUG
    printf("\n\n\n");
    printf("nv2 = %i\n", nv2);
    printf("na2 = %i\n", na2);
    printf("mem_size_V2 = %i\n", mem_size_V2);
    printf("mem_size_A2 = %i\n", mem_size_A2);
    mostrarUI(v2, nv2+1, "v2");
    mostrarUI(a2, na2, "a2");
#endif //_DEBUG
*/

    // cleanup memory
    //for (l=0; l<nv1; l++) {
    //    delete grafo_invertido[l];
    //}
    delete[] grafo_invertido;
}


//Invertir todos los grafos: v1, a1 y w1 se eliminan antes de construir v2, a2 y w2
void invertir_Grafos3(const unsigned int n_Megas, const unsigned int n_Grafos)
{
	//lISTA DE PRECDECESORES
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

    char s1[100];
    char s2[100];
    
    //DEPURACION
    printf("Invirtiendo los grafos\n\n");


    for (unsigned int m=12; m<=n_Megas; m++)
    {
        printf("\n\nMegas= %d\n\n", m);
        printf("Grafo\t Lectura\t Inversion\t Escritura\n\n");

        for (unsigned int i=2; i<=n_Grafos; i++)
        {
            printf("%i\t", i);

            if(i<10) sprintf(s1,"graphs/%d/grafo0%d.gr", m, i);
            else sprintf(s1,"graphs/%d/grafo%d.gr", m, i);

            if(i<10) sprintf(s2,"inverted_graphs/%d/grafo0%d.gr", m, i);
            else sprintf(s2,"inverted_graphs/%d/grafo%d.gr", m, i);

            //if(i<10) sprintf_s(s1,"E:/Chus/DATA/GRAFOS/%d/grafo0%d.gr", m, i);
            //else sprintf_s(s1,"E:/Chus/DATA/GRAFOS/%d/grafo%d.gr", m, i);

            //if(i<10) sprintf_s(s2,"E:/Chus/DATA/GRAFOS INVERTIDOS/%d/grafo0%d.gr", m, i);
            //else sprintf_s(s2,"E:/Chus/DATA/GRAFOS INVERTIDOS/%d/grafo%d.gr", m, i);


            /* Updated timer code for CUDA 9 */

            cudaEvent_t timerStart, timerStop;
            float time;

            // start things
            cudaEventCreate(&timerStart);
            cudaEventCreate(&timerStop);
            cudaEventRecord(timerStart, 0);

            //Leer grafo
            readGraphFromFile(s1, nv1, na1, mem_size_V1, mem_size_A1, v1, a1, w1);
            nv1 = nv1-1; //descontar el tapon

            // end things
            cudaEventRecord(timerStop, 0);
            cudaEventElapsedTime(&time, timerStart, timerStop);
            cudaEventDestroy(timerStart);
            cudaEventDestroy(timerStop);
            printf("%.6f", time);


            // start things
            cudaEventCreate(&timerStart);
            cudaEventCreate(&timerStop);
            cudaEventRecord(timerStart, 0);


/*
#ifdef _DEBUG
    printf("nv1 = %i\n", nv1);
    printf("na1 = %i\n", na1);
    printf("mem_size_V1 = %i\n", mem_size_V1);
    printf("mem_size_A1 = %i\n", mem_size_A1);
    mostrarUI(v1, nv1+1, "v1");
    mostrarUI(a1, na1, "a1");
#endif //_DEBUG
*/

            //Invertir grafo: OJO v1, a1 y w1 se destruyen!
            invertir_Grafo3(nv1, na1, mem_size_V1, mem_size_A1, v1, a1, w1,
                            nv2, na2, mem_size_V2, mem_size_A2, v2, a2, w2);


            // end things
            cudaEventRecord(timerStop, 0);
            cudaEventElapsedTime(&time, timerStart, timerStop);
            cudaEventDestroy(timerStart);
            cudaEventDestroy(timerStop);
            printf("%.6f", time);


            // start things
            cudaEventCreate(&timerStart);
            cudaEventCreate(&timerStop);
            cudaEventRecord(timerStart, 0);

            writeGraphToFile(s2, nv2+1, na2, mem_size_V2, mem_size_A2, v2, a2, w2);

            // end things
            cudaEventRecord(timerStop, 0);
            cudaEventElapsedTime(&time, timerStart, timerStop);
            cudaEventDestroy(timerStart);
            cudaEventDestroy(timerStop);
            printf("%.6f", time);

            printf( "%\n");

            // cleanup memory
            free(v2);
            free(a2);
            free(w2);
        }
    }
}


//Leer un trozo
void leerTrozo(const char* filename, 
               const unsigned int mem_size_trozo_V, const unsigned int mem_size_trozo_A, 
               const unsigned int offsetV, const unsigned int offsetA, 
               unsigned int*& v, unsigned int*& a, unsigned int*& w)
{
    ifstream entrada;
    entrada.open(filename, ios::binary);
    unsigned int* aux;

    if (entrada.fail()) {
        printf("\nProblemas con el archivo\n");
    } else {
        aux = &v[offsetV];
        entrada.read((char*)aux, mem_size_trozo_V);
        aux = &a[offsetA];
        entrada.read((char*)aux, mem_size_trozo_A);
        aux = &w[offsetA];
        entrada.read((char*)aux, mem_size_trozo_A);
        entrada.close();
    }	
}


//Guardar un trozo
void guardarTrozo(const char* filename, 
                  const unsigned int mem_size_trozo_V, const unsigned int mem_size_trozo_A, 
                  const unsigned int* v, const unsigned int* a, const unsigned int* w)
{
    ofstream salida;
    salida.open(filename, ios::binary);

    if (salida.fail()) {
        printf("\nProblemas con el archivo\n");
    } else {
        salida.write((char*)v, mem_size_trozo_V);
        salida.write((char*)a, mem_size_trozo_A);
        salida.write((char*)w, mem_size_trozo_A);
        salida.close();
    }
}


//Invertir un grafo troceando los arrays a2 y w2
void invertir_Grafo4(const unsigned int nv1, const unsigned int na1, 
                     const unsigned int mem_size_V1, const unsigned int mem_size_A1,
                     unsigned int* v1, unsigned int* a1, unsigned int* w1,
                     const unsigned int nTrozos, const unsigned int degree,
                     unsigned int& nv2, unsigned int& na2, 
                     unsigned int& mem_size_V2, unsigned int& mem_size_A2,
                     unsigned int*& v2, unsigned int*& a2, unsigned int*& w2)
{
	unsigned int trozo;
	//Se supone que nv1 es múltiplo de nTrozos
	unsigned int tam= nv1/nTrozos; 

	unsigned int offsetV;
	unsigned int offsetA;
	unsigned int sup;
	unsigned int i;

	char s[100];
	unsigned int* aristas= (unsigned int*) malloc(nTrozos*sizeof(unsigned int));
	
	unsigned int* index= (unsigned int*) malloc(tam*sizeof(unsigned int));

	unsigned int mem_size_trozo_V= tam*sizeof(unsigned int);
	v2= (unsigned int*) malloc(mem_size_trozo_V);

	unsigned int mem_size_trozo_A;
	
	unsigned int origen;
	unsigned int destino;
	unsigned int peso;

	unsigned int cont;
	unsigned int aux;

	offsetA= 0;
	for(trozo=0; trozo<nTrozos; trozo++){
		offsetV= trozo*tam;
		sup= offsetV+tam;

		//Inicializamos index
		for(i= 0; i<tam; i++) index[i]= 0;

		//Contamos el número de adyacentes
		for(destino= 0; destino<nv1; destino++){
			for(unsigned int d= v1[destino]; d<v1[destino+1]; d++){
				//arista destino <- a1[d]
				origen= a1[d];
				if((offsetV<=origen)&&(origen<sup)){
					origen= origen-offsetV;
					index[origen]= index[origen]+1;
				}//if
			}//for d
		}//for destino

		//calculamos v2 
		cont= 0;
		for(i= 0; i<tam; i++){
			v2[i]= offsetA+cont;
			aux= index[i];
			index[i]= cont;
			cont= cont+aux;
		}

		//anotamos el numero de aristas del trozo y actualizamos offsetA
		aristas[trozo]= cont;
		offsetA= offsetA+aristas[trozo];
		
		//reservamos espacio para a y w2
		mem_size_trozo_A= aristas[trozo]*sizeof(unsigned int);
	    a2= (unsigned int*) malloc(mem_size_trozo_A);
	    w2= (unsigned int*) malloc(mem_size_trozo_A);


		//Invertimos parte del grafo
		for(destino= 0; destino<nv1; destino++){
			for(unsigned int d= v1[destino]; d<v1[destino+1]; d++){
				//arista destino <- a1[d]
				origen= a1[d];
				if((offsetV<=origen)&&(origen<sup)){
					origen= origen-offsetV;
					peso= w1[d];
					a2[index[origen]]= destino;
					w2[index[origen]]= peso;
					index[origen]= index[origen]+1;
				}//if
			}//for d
		}//for destino

		//Grabar el trozo en fichero 
		if (trozo<10) {
                    sprintf(s,"./temp/trozo0%d.gr", trozo);
		} else {
                    sprintf(s,"./temp/trozo%d.gr", trozo);
                }

	        guardarTrozo(s, mem_size_trozo_V, mem_size_trozo_A, v2, a2, w2);

		free(a2);
	        free(w2);

	}//for trozo
	
	// cleanup memory, antes de construir v2, a2 y w2
	free( v1);
	free( a1);
	free( w1);

	free(index);
	free( v2);


	//JUNTANDO LOS TROZOS

	//RECUERDA: mem_size_V1= (nv1+1)*sizeof(int)
	nv2= nv1;
	na2= na1;
	mem_size_V2= mem_size_V1;
	mem_size_A2= mem_size_A1;

	v2= (unsigned int*) malloc(mem_size_V2);
	a2= (unsigned int*) malloc(mem_size_A2);
	w2= (unsigned int*) malloc(mem_size_A2);

	//incluir tapón
	v2[nv2]= na2;

	offsetA= 0;
	for(trozo=0; trozo<nTrozos; trozo++){
		offsetV= trozo*tam;
		mem_size_trozo_A= aristas[trozo]*sizeof(unsigned int);

		//leer trozo
		if(trozo<10) sprintf(s,"./temp/trozo0%d.gr", trozo);
		else sprintf(s,"./temp/trozo%d.gr", trozo);
		leerTrozo(s, mem_size_trozo_V, mem_size_trozo_A, offsetV, offsetA, v2, a2, w2);

		offsetA= offsetA + aristas[trozo];

	}//for trozo

	free(aristas);


/*
#ifdef _DEBUG
    printf("\n\n\n");
    printf("nv2 = %i\n", nv2);
    printf("na2 = %i\n", na2);
    printf("mem_size_V2 = %i\n", mem_size_V2);
    printf("mem_size_A2 = %i\n", mem_size_A2);
    mostrarUI(v2, nv2+1, "v2");
    mostrarUI(a2, na2, "a2");
#endif //_DEBUG
*/

}


//Invertir todos los grafos: v1, a1 y w1 se eliminan antes de construir v2, a2 y w2
void invertir_Grafos4(const unsigned int n_Megas, const unsigned int n_Grafos,
                      const unsigned int nTrozos, const unsigned int degree)
{

	//lISTA DE PRECDECESORES
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

    char s1[100];
    char s2[100];
    
    //DEPURACION
    printf("Invirtiendo los grafos\n\n");

    for (unsigned int m=12; m<=n_Megas; m++)
    {
        printf("\n\nMegas= %d\n\n", m);
        printf("Grafo\t Lectura\t Inversion\t Escritura\n\n");

        for (unsigned int i=2; i<=n_Grafos; i++)
        {
            printf("%i\t", i);

            if(i<10) sprintf(s1,"graphs/%d/grafo0%d.gr", m, i);
            else sprintf(s1,"graphs/%d/grafo%d.gr", m, i);

            if(i<10) sprintf(s2,"inverted_graphs/%d/grafo0%d.gr", m, i);
            else sprintf(s2,"inverted_graphs/%d/grafo%d.gr", m, i);

            //if(i<10) sprintf_s(s1,"E:/Chus/DATA/GRAFOS/%d/grafo0%d.gr", m, i);
            //else sprintf_s(s1,"E:/Chus/DATA/GRAFOS/%d/grafo%d.gr", m, i);

            //if(i<10) sprintf_s(s2,"E:/Chus/DATA/GRAFOS INVERTIDOS/%d/grafo0%d.gr", m, i);
            //else sprintf_s(s2,"E:/Chus/DATA/GRAFOS INVERTIDOS/%d/grafo%d.gr", m, i);


            /* Updated timer code for CUDA 9 */

            cudaEvent_t timerStart, timerStop;
            float time;

            // start things
            cudaEventCreate(&timerStart);
            cudaEventCreate(&timerStop);
            cudaEventRecord(timerStart, 0);

            //Leer grafo
            readGraphFromFile(s1, nv1, na1, mem_size_V1, mem_size_A1, v1, a1, w1);
            nv1 = nv1-1; //descontar el tapon

            // end things
            cudaEventRecord(timerStop, 0);
            cudaEventElapsedTime(&time, timerStart, timerStop);
            cudaEventDestroy(timerStart);
            cudaEventDestroy(timerStop);
            printf("%.6f", time);


            // start things
            cudaEventCreate(&timerStart);
            cudaEventCreate(&timerStop);
            cudaEventRecord(timerStart, 0);

/*
#ifdef _DEBUG
    printf("nv1 = %i\n", nv1);
    printf("na1 = %i\n", na1);
    printf("mem_size_V1 = %i\n", mem_size_V1);
    printf("mem_size_A1 = %i\n", mem_size_A1);
    mostrarUI(v1, nv1+1, "v1");
    mostrarUI(a1, na1, "a1");
#endif //_DEBUG
*/

            //Invertir grafo: OJO v1, a1 y w1 se destruyen!
            invertir_Grafo4(nv1, na1, mem_size_V1, mem_size_A1, v1, a1, w1, nTrozos, degree,
                            nv2, na2, mem_size_V2, mem_size_A2, v2, a2, w2);

            // end things
            cudaEventRecord(timerStop, 0);
            cudaEventElapsedTime(&time, timerStart, timerStop);
            cudaEventDestroy(timerStart);
            cudaEventDestroy(timerStop);
            printf("%.6f", time);


            // start things
            cudaEventCreate(&timerStart);
            cudaEventCreate(&timerStop);
            cudaEventRecord(timerStart, 0);
	
            writeGraphToFile(s2, nv2+1, na2, mem_size_V2, mem_size_A2, v2, a2, w2);

            // end things
            cudaEventRecord(timerStop, 0);
            cudaEventElapsedTime(&time, timerStart, timerStop);
            cudaEventDestroy(timerStart);
            cudaEventDestroy(timerStop);
            printf("%.6f", time);

            printf( "%\n");

            // cleanup memory
            free(v2);
            free(a2);
            free(w2);
        }
    }
}


//Imprimir degrees de un grafo//Imprimir degrees de un grafo
void imprimir_Degrees(const unsigned int nv, const unsigned int* v)
{
	int maximo= v[1]-v[0];
	unsigned int i_max= 0;
	int minimo= v[1]-v[0];
	unsigned int i_min= 0;

	int d;
	for(unsigned int i=1; i<nv; i++){ 
		d= v[i+1]-v[i];
		//printf("%d\t", d);
		if(maximo<d){
			maximo= d;
			i_max= i;
		}
		if(minimo>d){
			minimo= d;
			i_min= i;
		}
	}
	//printf("\n\n");
	//printf("Minimo= %d\t\tMaximo= %d\n\n", minimo, maximo);
	printf("%d\t%d\t%d\t%d\t%d\t%d", minimo, i_min, v[i_min], maximo, i_max, v[i_max]);

}


//Imprimir degrees de todos los grafos
void imprimir_Degrees_Grafos(const unsigned int n_Megas, const unsigned int n_Grafos) {
    //lISTA DE ADYACENTES
    unsigned int* v; //array de vértices host
    unsigned int nv; //número de vértices 
    unsigned int mem_size_V; //memoria del array con tapon

    unsigned int* a; //array de aristas host
    unsigned int na; //número de aristas
    unsigned int mem_size_A; //memoria del array

    unsigned int* w; //array de pesos host


    char s[100];
    
    //DEPURACION
    printf("DEGREE DE LOS GRAFOS INVERTIDOS\n\n");

    for (unsigned int m=1; m<=n_Megas; m++)
    {
        printf("\n\nMegas= %d\n\n", m);
        printf("Grafo\tMinimo\ti_min\v[ti_min]\tMaximo\ti_max\tv[ti_max]\n");

        for (unsigned int i=1; i<=n_Grafos; i++)
        {
            printf("%i\t",i);

            if (i<10) {
                sprintf(s,"inverted_graphs/%d/grafo0%d.gr", m, i);
            } else {
                sprintf(s,"inverted_graphs/%d/grafo%d.gr", m, i);
            }

            //Leer grafo
            readGraphFromFile(s, nv, na, mem_size_V, mem_size_A, v, a, w);
            nv = nv-1; //descontar el tapon

            mostrarUI(v, nv+1, "v");

            //Imprimir degrees
            imprimir_Degrees(nv, v);

            printf("\n");

            // cleanup memory
            free(v);
            free(a);
            free(w);
        }
    }
}


//Test para probar operaciones sobre grafos
void test_Grafos(unsigned int nv, unsigned int degree, unsigned int topeW)
{
    unsigned int* v1;           //array de vértices host
    unsigned int nv1 = nv;
    unsigned int mem_size_V1;   //memoria del array con tapon

    unsigned int* a1;           //array de aristas host
    unsigned int na1;           //número de aristas
    unsigned int mem_size_A1;   //memoria del array

    unsigned int* w1;           //array de pesos host

    unsigned int infinito = nv * topeW;


    //DEPURACION
    printf("Generando Grafo\n\n");

    generaGrafo(nv1, degree, topeW, mem_size_V1, na1, mem_size_A1, v1, a1, w1);

    // cleanup memory
    free(v1);
    free(a1);
    free(w1);
}


//Generación de grafos
void generar_Grafos(const unsigned int n_Megas, const unsigned int n_Grafos,
                    unsigned int degree, unsigned int topeW)
{
    unsigned int* v;            //array de vértices host
    unsigned int nv;            //número de vértices 
    unsigned int mem_size_V;    //memoria del array con tapon

    unsigned int* a;            //array de aristas host
    unsigned int na;            //número de aristas
    unsigned int mem_size_A;    //memoria del array

    unsigned int* w;            //array de pesos host

    char s[100];

    //DEPURACION
    printf("Generando y Guardando Grafos\n\n");

    //for (unsigned int m=9; m<=14; m++)

    for (unsigned int m=11; m<=11; m++)
    {
        printf("\n\nMegas = %d\n", m);
        //nv = m*1024*1024;
        nv = m*1024;

        printf("Grafo\t Generar\t Salvar\n");

       	//for (unsigned int i=1; i<=25; i++)
       	for (unsigned int i=1; i<=5; i++)
        {
            printf("%i\t", i);

            if (i<10) {
                //sprintf(s,"graphs/%d/grafo0%d.gr", m, i);
                sprintf(s,"graphs/grafo-%d-0%d.gr", m, i);
            } else {
                //sprintf(s,"graphs/%d/grafo%d.gr", m, i);
                sprintf(s,"graphs/grafo-%d-%d.gr", m, i);
            }

            /* Updated timer code for CUDA 9 */

            cudaEvent_t timerStart, timerStop;
            float time;

            // start things
            cudaEventCreate(&timerStart);
            cudaEventCreate(&timerStop);
            cudaEventRecord(timerStart, 0);

            generaGrafo(nv, degree, topeW, mem_size_V, na, mem_size_A, v, a, w);

            // end things
            cudaEventRecord(timerStop, 0);
            cudaEventElapsedTime(&time, timerStart, timerStop);
            cudaEventDestroy(timerStart);
            cudaEventDestroy(timerStop);
            printf("%.6f", time);


            // start things
            cudaEventCreate(&timerStart);
            cudaEventCreate(&timerStop);
            cudaEventRecord(timerStart, 0);

	    writeGraphToFile(s, nv+1, na, mem_size_V, mem_size_A, v, a, w);

            // end things
            cudaEventRecord(timerStop, 0);
            cudaEventElapsedTime(&time, timerStart, timerStop);
            cudaEventDestroy(timerStart);
            cudaEventDestroy(timerStop);
            printf("%.6f", time);


            // cleanup memory
            free(v);
            free(a);
            free(w);
        }
    }
}


