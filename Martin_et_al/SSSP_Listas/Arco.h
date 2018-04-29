//---------------------------------------------------------------------------

#ifndef ArcoH
#define ArcoH

class Arco{
private: 
	unsigned int destino;
	unsigned int peso;

public: 
	Arco(unsigned int d, unsigned int p);
	~Arco();
	unsigned int getDestino();
	unsigned int getPeso();
};

#endif //ArcoH
