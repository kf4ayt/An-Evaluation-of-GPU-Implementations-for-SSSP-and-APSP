//---------------------------------------------------------------------------

#ifndef NodoH
#define NodoH
//---------------------------------------------------------------------------
class Nodo{
private:
   unsigned int destino;
   unsigned int peso;
   Nodo* sig;

public:
   Nodo(unsigned int d1, unsigned int p1, Nodo* s1);
   ~Nodo();
   Nodo* getSig();
   unsigned int getDestino();
   unsigned int getPeso();
};


#endif //NodoH
