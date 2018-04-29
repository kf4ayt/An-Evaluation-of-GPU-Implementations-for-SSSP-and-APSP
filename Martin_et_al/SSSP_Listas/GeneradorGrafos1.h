// export C interface

////////////////////////////////
// LECTURA Y ESCRITURA DE GRAFOS
////////////////////////////////
extern "C" 
void guardaGrafo_FicheroB(const char* filename, const unsigned int nv, const unsigned int na,
						  const unsigned int mem_size_V, const unsigned int mem_size_A,
						  const unsigned int* v, const unsigned int* a, const unsigned int* w);

extern "C" 
void leeGrafo_FicheroB(const char* filename, 
				      unsigned int& nv, unsigned int& na, 
					  unsigned int& mem_size_V, unsigned int& mem_size_A,
					  unsigned int*& v, unsigned int*& a, unsigned int*& w);

//////////////////////
//GENERACIÓN DE GRAFOS
//////////////////////
extern "C" 
unsigned int RangedRand( unsigned int range_min, unsigned int range_max);

extern "C" 
void generaGrafo( unsigned int nv, const unsigned int degree, 
				   const unsigned int topeW,  unsigned int&  mem_size_V,
		     	   unsigned int& na, unsigned int& mem_size_A,
				   unsigned int*& v, unsigned int*& a, unsigned int*& w);

extern "C" 
void generar_Grafos(const unsigned int n_Megas, const unsigned int n_Grafos,
					unsigned int degree, unsigned int topeW);

/////////////////////
//INVERSION DE GRAFOS
/////////////////////
extern "C" 
void invertir_Grafo(const unsigned int nv1, const unsigned int na1, 
					const unsigned int mem_size_V1, const unsigned int mem_size_A1,
					const unsigned int* v1, const unsigned int* a1, const unsigned int* w1,
					unsigned int& nv2, unsigned int& na2, 
					unsigned int& mem_size_V2, unsigned int& mem_size_A2,
					unsigned int*& v2, unsigned int*& a2, unsigned int*& w2);

extern "C" 
void invertir_Grafos(const unsigned int n_Megas, const unsigned int n_Grafos);


////////////////////////////////////////////////////////////
//OPERACIONES SOBRE GRAFOS
////////////////////////////////////////////////////////////
extern "C" 
void imprimir_Degrees( const unsigned int nv, const unsigned int* v);

extern "C" 
void imprimir_Degrees_Grafos( const unsigned int n_Megas, const unsigned int n_Grafos);

extern "C" 
void test_Grafos(const unsigned int nv, const unsigned int degree, const unsigned int topeW);

