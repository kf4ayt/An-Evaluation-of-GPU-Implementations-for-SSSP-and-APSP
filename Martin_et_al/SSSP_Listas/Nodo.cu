#include "Nodo.h"

Nodo::Nodo(unsigned int d1, unsigned int p1, Nodo* s1){
	sig=s1;
	destino=d1;
	peso=p1;
}

Nodo::~Nodo(){
}

Nodo* Nodo::getSig(){
	return sig;
}

unsigned int Nodo::getDestino(){
	return destino;
}

unsigned int Nodo::getPeso(){
	return peso;
}
