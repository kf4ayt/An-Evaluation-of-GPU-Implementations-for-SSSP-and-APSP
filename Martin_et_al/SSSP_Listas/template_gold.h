#ifndef _TEMPLATE_GOLD
#define _TEMPLATE_GOLD

void computeGold_SSSP3(unsigned int* reference, 
                       const unsigned int nv, const unsigned int* v, 
                       const unsigned int na, const unsigned int* a, const unsigned int* w, 
                       const unsigned int infinito);

void computeGold_SSSP8(unsigned int* reference, 
                       const unsigned int nv, const unsigned int* v, 
                       const unsigned int na, const unsigned int* a, const unsigned int* w, 
                       const unsigned int infinito);

void computeGold_Dijkstra(unsigned int* reference, 
                          const unsigned int nv, const unsigned int* v, 
                          const unsigned int na, const unsigned int* a, const unsigned int* w, 
                          const unsigned int infinito);

#endif // #ifndef _TEMPLATE_GOLD
