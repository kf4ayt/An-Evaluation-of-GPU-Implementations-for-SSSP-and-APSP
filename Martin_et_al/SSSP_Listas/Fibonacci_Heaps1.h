// export C interface

extern "C" 
void computeGold_FH(unsigned int* reference, 
                    const unsigned int nv, const unsigned int* v, 
                    const unsigned int na, const unsigned int* a, const unsigned int* w, 
                    const unsigned int infinito, f_heap* fh, heap_node** hn);
