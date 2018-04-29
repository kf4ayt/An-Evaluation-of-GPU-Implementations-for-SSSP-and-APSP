
#include "Fibonacci_Heaps.h"
//#include "Fibonacci_Heaps1.h"

#include "tools.h"

#include <stdio.h>
#include <stdlib.h>
#include <math.h>


/* CWJ includes */

#include <cuda.h>


/*******************   functions for F-heap  *****************/




heap_node    *after, *before, *father, *child, *first, *last,
	         *node_c, *node_s, *node_r, *node_n, *node_l;

unsigned int  dg;

#define BASE           1.61803 //Número de oro

#define OUT_OF_HEAP    0
#define IN_HEAP        1
#define MARKED         2


void Init_fheap(f_heap* fh, const unsigned int nv, const unsigned int* reference1)
{
    fh->deg_max = (unsigned int) (log((double)nv) / log(BASE) + 1 );

    if ((fh->deg_pointer = (heap_node**) malloc (fh->deg_max* sizeof(heap_node*)))
	==
	(heap_node**)NULL)
    {
        exit(NOT_ENOUGH_MEM);
    }

    for (dg = 0; dg < fh->deg_max; dg++) {
        fh->deg_pointer[dg] = HNNULL;
    }

    fh->n = 0;
    fh->min = HNNULL;
    fh->reference = reference1;
}

void Check_min(f_heap* fh, heap_node* nd)
{
    if (fh->reference[nd->vertex_index] < fh->dist) { //reference
        fh->dist = fh->reference[nd->vertex_index];   //reference
        fh->min  = nd;
    }
}

void Insert_after_min(f_heap* fh, heap_node* nd)
{
    after = fh->min->next;
    nd->next = after;
    after->prev = nd;
    fh->min->next = nd;
    nd->prev = fh->min;

    Check_min(fh, nd);
}

void Insert_to_root(f_heap* fh, heap_node* nd)
{
    nd->heap_parent = HNNULL;
    nd->status      = IN_HEAP; //status

    Insert_after_min(fh, nd);
}

void Insert_to_fheap( f_heap* fh, heap_node* nd)
{
    nd->heap_parent = HNNULL;
    nd->son         = HNNULL;
    nd->status      = IN_HEAP; //status
    nd->deg         = 0;

    if (fh->min == HNNULL )
    {
        nd->prev = nd->next = nd;
        fh->min = nd;
        fh->dist = fh->reference[nd->vertex_index]; //reference
    } else {
        Insert_after_min(fh, nd);
    }

    fh->n++;
}

void Cut_node(f_heap* fh, heap_node* nd, heap_node* father)
{
    after = nd->next;

    if (after != nd) { 
        before = nd->prev;
        before->next = after;
        after->prev = before;
    }

    if ( father -> son == nd ) father -> son = after;
    ( father -> deg ) --;
    if ( father -> deg == 0 ) father -> son = HNNULL;
}

void Fheap_decrease_key(f_heap* fh, heap_node* nd)
{
    if ((father = nd->heap_parent) == HNNULL)
    {
        Check_min (fh, nd);
    }
    else /* node isn't in the root */ 
    {
        if (fh->reference[nd->vertex_index] < fh->reference[father->vertex_index]) { //reference
            node_c = nd;

            while (father != HNNULL){
                Cut_node(fh, node_c, father);
                Insert_to_root(fh, node_c);

                if (father->status == IN_HEAP) {
                    father->status = MARKED;
                    break;
                }

                node_c = father;
                father =  father->heap_parent;
            }
        }
    }
}

heap_node* Extract_min(f_heap* fh, const unsigned int infinito)
{
	heap_node *nd;
	nd = fh->min;
	if ( fh->n > 0 ){
		fh->n --;
		fh->min -> status = OUT_OF_HEAP; //status

		/* connecting root-list and sons-of-min-list */ 
		first = fh->min -> prev;
		child = fh->min -> son;
		if ( first == fh->min )
			first = child;
		else{
			after = fh->min -> next;
			if ( child == HNNULL ){
				first -> next = after;
				after -> prev = first;
			}//if
			else{
				before = child -> prev;
				first  -> next = child;
				child  -> prev = first;

				before -> next = after;
				after  -> prev = before;
			}//else
		}//else
		
		if ( first != HNNULL ){ /* heap is not empty */ 
			/* squeezing root */ 
			node_c = first;
			last   = first -> prev;
			while(1){
				node_l = node_c;
				node_n = node_c -> next;
				/*    printf ( "node_c=%ld  node_n=%ld\n", nod(node_c), nod(node_n) );*/
				while(1){
					dg = node_c -> deg;
					node_r = fh->deg_pointer[dg];
					/*
					printf ( "dg=%ld  node_r=%ld\n", dg, nod(node_r) );
					for ( dgx = 0; dgx < fh->deg_max; dgx ++ )
						printf (" %ld ", nod(fh->deg_pointer[dgx]) );
					printf ("\n");
					*/
					if( node_r == HNNULL ){
						fh->deg_pointer[dg] = node_c;
						break;
					}//if
					else{
						if(fh->reference[node_c->vertex_index] < fh->reference[node_r->vertex_index]){
							node_s = node_r;
							node_r = node_c;
						}//if
						else
							node_s = node_c;

						/*    printf ( "node_r=%ld  node_s=%ld\n", nod(node_r), nod(node_s) );*/
						/* detach node_s from root */ 
						after  = node_s -> next;
						before = node_s -> prev;

						after  -> prev = before;
						before -> next = after;

						/* attach node_s to node_r */ 
						node_r -> deg ++;
						node_s -> heap_parent = node_r;
						node_s -> status = IN_HEAP;

						child = node_r -> son;

						if ( child == HNNULL )
							node_r -> son = node_s -> next = node_s -> prev = node_s;
						else{
							after = child -> next;
							child  -> next = node_s;
							node_s -> prev = child;
							node_s -> next = after;
							after  -> prev = node_s;
						}//else
					} 
					
					/* node_r now is father of node_s */ 
					node_c = node_r;
					fh->deg_pointer[dg] = HNNULL;
					/*
					printf ( "INHEAP node  dist parent  son next prev deg status\n" );
					for ( i = nodes; i != node_last; i ++ ){
						if ( i -> status > OUT_OF_HEAP )
							printf (" %ld %ld %ld %ld  %ld %ld %ld %d\n",
									nod(i), i->dist, nod(i->heap_parent), nod(i->son), nod(i->next),
									nod(i->prev), i->deg, i->status );
					}
					fgetc(inp);
					*/

				}//while
				
				if ( node_l == last ) break;
				node_c = node_n;
			}//while
			
			/* finding minimum */ 
			fh->dist = infinito;

		    for ( dg = 0; dg < fh->deg_max; dg ++ ){
				if ( fh->deg_pointer[dg] != HNNULL ){
					node_r = fh->deg_pointer[dg];
					fh->deg_pointer[dg] = HNNULL;
					Check_min ( fh, node_r );
					node_r -> heap_parent = HNNULL;
				}//if
			}//for
		}//if
		else /* heap is empty */ 
			fh->min = HNNULL;
	}//if
	return nd;
}


/**************   end of heap functions   ****************/

#define NODE_IN_FHEAP( node ) ( node -> status > OUT_OF_HEAP )

void init_HN(heap_node* nd, const unsigned int index)
{
    nd->vertex_index = index;
    nd->heap_parent = HNNULL;
    nd->son = HNNULL;
    nd->next = HNNULL;
    nd->prev = HNNULL;
    nd->deg = 0;
    nd->status = OUT_OF_HEAP;
}

void computeGold_FH(unsigned int* reference,
                    const unsigned int nv, const unsigned int* v, 
                    const unsigned int na, const unsigned int* a, const unsigned int* w, 
                    const unsigned int infinito, f_heap* fh, heap_node** hn)
{
    //INICIALIZACION
    unsigned int i;
    init_HN(hn[0], 0);
    reference[0] = 0;

    for (i=1; i<nv; i++) {
        init_HN(hn[i], i);
        reference[i] = infinito;
    }	


    /* Updated timer code for CUDA 9 */

    cudaEvent_t timerStart, timerStop;
    float time;

    //Bucle
    Init_fheap(fh, nv, reference);
    Insert_to_fheap(fh, hn[0]);
    unsigned int nVueltas= 0;
    heap_node* nd;
    unsigned int frontera;
    unsigned int dist_new;
    unsigned int sid;

    cudaEventCreate(&timerStart);
    cudaEventCreate(&timerStop);
    cudaEventRecord(timerStart, 0);

    while (true) {
        nVueltas++;		
		
        //Cálculo de la frontera
        nd = Extract_min(fh, infinito);

        if (nd == HNNULL) {
            break;
        }

        frontera = nd->vertex_index;

        //Relajamos
        for (i=v[frontera]; i<v[frontera+1]; i++)
        {
            sid= a[i];
            dist_new = reference[frontera] + w[i];

            if (dist_new < reference[sid])
            {
                reference[sid] = dist_new;

                if (NODE_IN_FHEAP(hn[sid])) {
                    Fheap_decrease_key(fh, hn[sid]);
                } else {
                    Insert_to_fheap(fh, hn[sid]);
                }
            }
        }
    }
	
    cudaEventRecord(timerStop, 0);
    cudaEventSynchronize(timerStop);
    
    cudaEventElapsedTime(&time, timerStart, timerStop);
    cudaEventDestroy(timerStart);
    cudaEventDestroy(timerStop);

    printf("Runtime for computeGold_FH algorithm is: %.6f ms\n", time);

    //destrucción del array fh->deg_pointer
    free(fh->deg_pointer);


#ifdef _DEBUG
    //mostrarUI(reference,nv,"reference FH");
#endif //_DEBUG

}


