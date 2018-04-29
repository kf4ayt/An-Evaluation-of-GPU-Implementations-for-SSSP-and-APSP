
#ifndef _Fibonacci_Heaps
#define _Fibonacci_Heaps

typedef  /* heap_node */
   struct heap_node_st
{
   unsigned int			  vertex_index;	   //índice al vértice
   struct heap_node_st   *heap_parent;     /* heap parent pointer */
   struct heap_node_st   *son;             /* heap successor */
   struct heap_node_st   *next;            /* next brother */
   struct heap_node_st   *prev;            /* previous brother */
   unsigned int           deg;             /* number of children */
   unsigned int           status;          /* status of node */
} heap_node;

#define HNNULL	        (heap_node*)NULL

typedef /* F-heap */
   struct fheap_st
{
   heap_node            *min;        /* the minimal node */
   unsigned int         dist;        /* tentative distance of min. node */
   unsigned int            n;        /* number of nodes in the heap */
   heap_node   **deg_pointer;        /* pointer to the node with given degree */
   unsigned int      deg_max;        /* maximal degree */

   const unsigned int*    reference; //La actual estimacion de todos los vértices
}
   f_heap;


void computeGold_FH(unsigned int* reference, 
                    const unsigned int nv, const unsigned int* v, 
                    const unsigned int na, const unsigned int* a, const unsigned int* w, 
                    const unsigned int infinito, f_heap* fh, heap_node** hn);

#endif //_Fibonacci_Heaps
