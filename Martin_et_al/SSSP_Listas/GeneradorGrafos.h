

/*******

The code below is the original code, edited so that it would run on CUDA
Compute Capability 6.1 hardware (EVGA/NVIDIA GTX 1070) with CUDA v9.0.176.
The display driver being used is NVIDIA 384.111. The OS is Debian Linux v9
('Sid').

Charles W Johnson
April, 2018

*******/


#ifndef _GENERADORGRAFOS
#define _GENERADORGRAFOS

using namespace std;


////////////////////////////////
// LECTURA Y ESCRITURA DE GRAFOS
////////////////////////////////

void guardaGrafo_FicheroB(const char* filename, const unsigned int nv, const unsigned int na,
                          const unsigned int mem_size_V, const unsigned int mem_size_A,
                          const unsigned int* v, const unsigned int* a, const unsigned int* w);


void writeGraphToFile(const char* filename, const unsigned int nv, const unsigned int na,
                      const unsigned int mem_size_V, const unsigned int mem_size_A,
                      const unsigned int* v, const unsigned int* a, const unsigned int* w);


void leeGrafo_FicheroB(const char* filename, 
                       unsigned int& nv, unsigned int& na, 
                       unsigned int& mem_size_V, unsigned int& mem_size_A,
                       unsigned int*& v, unsigned int*& a, unsigned int*& w);

//////////////////////
//GENERACIÓN DE GRAFOS
//////////////////////

unsigned int RangedRand(unsigned int range_min, unsigned int range_max);

void generaGrafo(unsigned int nv, const unsigned int degree, 
                 const unsigned int topeW,  unsigned int&  mem_size_V,
                 unsigned int& na, unsigned int& mem_size_A,
                 unsigned int*& v, unsigned int*& a, unsigned int*& w);


void generar_Grafos(const unsigned int n_Megas, const unsigned int n_Grafos,
                    unsigned int degree, unsigned int topeW);

/////////////////////
//INVERSION DE GRAFOS
/////////////////////
void invertir_Grafo(const unsigned int nv1, const unsigned int na1, 
                    const unsigned int mem_size_V1, const unsigned int mem_size_A1,
                    const unsigned int* v1, const unsigned int* a1, const unsigned int* w1,
                    unsigned int& nv2, unsigned int& na2, 
                    unsigned int& mem_size_V2, unsigned int& mem_size_A2,
                    unsigned int*& v2, unsigned int*& a2, unsigned int*& w2);

void invertir_Grafo2(const unsigned int nv1, const unsigned int na1, 
                     const unsigned int mem_size_V1, const unsigned int mem_size_A1,
                     unsigned int* v1, unsigned int* a1, unsigned int* w1,
                     unsigned int& nv2, unsigned int& na2, 
                     unsigned int& mem_size_V2, unsigned int& mem_size_A2,
                     unsigned int*& v2, unsigned int*& a2, unsigned int*& w2);

void invertir_Grafo3(const unsigned int nv1, const unsigned int na1, 
                     const unsigned int mem_size_V1, const unsigned int mem_size_A1,
                     unsigned int* v1, unsigned int* a1, unsigned int* w1,
                     unsigned int& nv2, unsigned int& na2, 
                     unsigned int& mem_size_V2, unsigned int& mem_size_A2,
                     unsigned int*& v2, unsigned int*& a2, unsigned int*& w2);

void invertir_Grafo4(const unsigned int nv1, const unsigned int na1, 
                     const unsigned int mem_size_V1, const unsigned int mem_size_A1,
                     unsigned int* v1, unsigned int* a1, unsigned int* w1,
                     const unsigned int nTrozos, const unsigned int degree,
                     unsigned int& nv2, unsigned int& na2, 
                     unsigned int& mem_size_V2, unsigned int& mem_size_A2,
                     unsigned int*& v2, unsigned int*& a2, unsigned int*& w2);

void invertir_Grafos(const unsigned int n_Megas, const unsigned int n_Grafos);
void invertir_Grafos2(const unsigned int n_Megas, const unsigned int n_Grafos);
void invertir_Grafos3(const unsigned int n_Megas, const unsigned int n_Grafos);
void invertir_Grafos4(const unsigned int n_Megas, const unsigned int n_Grafos,
                      const unsigned int nTrozos, const unsigned int degree);


////////////////////////////////////////////////////////////
//OPERACIONES SOBRE GRAFOS
////////////////////////////////////////////////////////////
void imprimir_Degrees(const unsigned int nv, const unsigned int* v);

void imprimir_Degrees_Grafos(const unsigned int n_Megas, const unsigned int n_Grafos);

void test_Grafos(unsigned int nv, unsigned int degree, unsigned int topeW);



#endif //#ifndef _GENERADORGRAFOS



