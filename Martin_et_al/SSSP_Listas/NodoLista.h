//---------------------------------------------------------------------------

#ifndef NodoListaH
#define NodoListaH
//---------------------------------------------------------------------------
template <class T> class NodoLista{
private:
   T* inf;
   NodoLista* sig;
   NodoLista* ant;

public:
   NodoLista(T* t, NodoLista<T>* sig1, NodoLista<T>* ant1);
   ~NodoLista();
   NodoLista<T>* getSig();
   NodoLista<T>* getAnt();
   void setSig(NodoLista<T>* n);
   void setAnt(NodoLista<T>* n);
   T* getInf();
};


template <class T> NodoLista<T>::
        NodoLista(T* t, NodoLista<T>* sig1, NodoLista<T>* ant1){
  inf= t;
  sig=sig1;
  ant=ant1;
}

template <class T> NodoLista<T>::~NodoLista(){
  delete inf;
}

template <class T> NodoLista<T>* NodoLista<T>::getSig(){
  return sig;
}

template <class T> NodoLista<T>* NodoLista<T>::getAnt(){
  return ant;
}

template <class T> T* NodoLista<T>::getInf(){
  return inf;
}

template <class T> void NodoLista<T>::setSig(NodoLista<T>* n){
  sig=n;
}

template <class T> void NodoLista<T>::setAnt(NodoLista<T>* n){
  ant=n;
}


#endif
