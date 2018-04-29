
#include "tools.h"

#include <stdlib.h>
#include <stdio.h>
#include <string.h>

#include <cuda.h>


//IMPLEMENTACION
void mostrarB(const bool* v, const unsigned int n, const char* vs){
    printf("\n%s[0..%i]\n", vs, n-1);    
    for( unsigned int i = 0; i < n; i++){
        printf( "%i\t", v[i]);
    }
    printf("\n\n");    
}
void mostrarI(const int* v, const unsigned int n, const char* vs){
    printf("\n%s[0..%i]\n", vs, n-1);    
    for( unsigned int i = 0; i < n; i++){
        printf( "%i\t", v[i]);
    }
    printf("\n\n");    
}
void mostrarUI(const unsigned int* v, const unsigned int n, const char* vs){
    printf("\n%s[0..%i]\n", vs, n-1);    
    for( unsigned int i = 0; i < n; i++){
        printf( "%i\t", v[i]);
    }
    printf("\n\n");    
}


unsigned int minimo(const unsigned int a, const unsigned int b){
	if(a<b) return a;
	return b;
}

