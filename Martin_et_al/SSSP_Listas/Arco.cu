//---------------------------------------------------------------------------
#include "Arco.h"

Arco::Arco(unsigned int d, unsigned int p) {
	destino = d;
	peso = p;
}

Arco::~Arco() {
}

unsigned int Arco::getDestino() {
	return destino;
}

unsigned int Arco::getPeso() {
	return peso;
}
