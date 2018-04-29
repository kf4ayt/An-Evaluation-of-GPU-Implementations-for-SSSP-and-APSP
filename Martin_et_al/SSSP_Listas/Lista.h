//---------------------------------------------------------------------------

#ifndef ListaH
#define ListaH

#include "NodoLista.h"
//---------------------------------------------------------------------------

template <class T> class Lista{
private:
   NodoLista<T>* primero;
   NodoLista<T>* ultimo;
   NodoLista<T>* actual;
   int longitud;

public:
   Lista();
   ~Lista();
   void inicia();
   void avanza();
   T* getActual();
   bool vacia();
   bool final();
   void insertaFinal(T*);
   void insertaPrincipio(T*);
   //void eliminaActual(bool borrar);
   int getLongitud();
};

template <class T> Lista<T>::Lista(){
  primero=NULL;
  ultimo=NULL;
  actual=NULL;
  longitud= 0;
}

template <class T> Lista<T>::~Lista(){
  NodoLista<T>* r= primero;
  NodoLista<T>* aux;
  while(r!=NULL){
    aux= r->getSig();
    delete r;
    r= aux;
  }
}

template <class T> void Lista<T>::inicia(){
  actual=primero;
}

template <class T> T* Lista<T>::getActual(){
  if(actual==NULL) return NULL;
  else return actual->getInf();
}

template <class T> void Lista<T>::avanza(){
  if(actual!=NULL) actual= actual->getSig();
}

template <class T> bool Lista<T>::vacia(){
  return primero==NULL;
}

template <class T> bool Lista<T>::final(){
  return actual==NULL;
}

template <class T> void Lista<T>::insertaFinal(T* t){
  if(vacia()){
    primero= new NodoLista<T>(t, NULL, NULL);
    actual=primero;
    ultimo=primero;
  }
  else{//Inserción por el final
    NodoLista<T>* aux= ultimo;
    ultimo= new NodoLista<T>(t,NULL,aux);
    aux->setSig(ultimo);
  }
  longitud++;
}

template <class T> void Lista<T>::insertaPrincipio(T* t){
  if(vacia()){
    primero= new NodoLista<T>(t, NULL, NULL);
    actual=primero;
    ultimo=primero;
  }
  else{//Inserción por el principio
    NodoLista<T>* aux= primero;
    primero= new NodoLista<T>(t, aux, NULL);
    aux->setAnt(primero);
  }
  longitud++;
}

/*
template <class T> void Lista<T>::eliminaActual(bool borrar){
  if(actual!=NULL){
    if(actual->getSig()!=NULL)
       actual->getSig()->setAnt(actual->getAnt());
    if(actual->getAnt()!=NULL)
       actual->getAnt()->setSig(actual->getSig());

    if(actual==primero) primero= primero->getSig();
    if(actual==ultimo) ultimo= ultimo->getAnt();
    
    NodoLista<T>* aux= actual->getSig();
    if(borrar) actual->limpia(); //borrar la información almacenada
    delete actual; //borrar el nodo
    actual= aux;
    longitud--;
  }
}
*/

template <class T> int Lista<T>::getLongitud(){
  return longitud;
}

#endif
