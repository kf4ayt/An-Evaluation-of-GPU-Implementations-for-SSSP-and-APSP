

/*******

The code below is the original code, edited so that it would run on CUDA
Compute Capability 6.1 hardware (EVGA/NVIDIA GTX 1070) with CUDA v9.0.176.
The display driver being used is NVIDIA 384.111. The OS is Debian Linux v9
('Sid').

Charles W Johnson
April, 2018

*******/


/*
KERNELS
*/


#ifndef _TEMPLATE_KERNEL_H_
#define _TEMPLATE_KERNEL_H_

#include <stdio.h>


/* CWJ includes */

#include <cuda.h>

#include "comun.cu"

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }


/////////////////////////////////////
/////////////////////////////////////
//SSSP0, SSSP1, SSSP2, SSSP3, y SSSP4
/////////////////////////////////////
/////////////////////////////////////

__global__ void kernel1(const unsigned int* v_d, const unsigned int* a_d, const unsigned int* w_d,
                        const bool* p_d, const bool* f_d, unsigned int* c_d)
{
    // access thread id
    const unsigned int tid = threadIdx.x + blockIdx.x * blockDim.x ;

    unsigned int i;
    unsigned int tope;
    unsigned int pid;

    if (p_d[tid]) {
        //visitando los predecesores de tid
        i = v_d[tid];
        tope = v_d[tid+1];

        while (i<tope) {
            pid = a_d[i];
            if (f_d[pid]) {
                if (c_d[tid] > (c_d[pid] + w_d[i])) {
                   c_d[tid] = c_d[pid] + w_d[i];
                } 
            }
            i++;
        }
    }
}



////////////////////////////////
__global__ void kernel3(bool* p_d, bool* f_d, 
                        const unsigned int* c_d, const unsigned int minimo)
{
    // access thread id
    const unsigned int tid = threadIdx.x + blockIdx.x * blockDim.x ;

    f_d[tid] = false;  

    if (p_d[tid] && (c_d[tid] == minimo)) {
        f_d[tid] = true;
        p_d[tid] = false;
    }
}


////////////////////////////////
__global__ void
kernel_minimizar1( const bool* p_d, const unsigned int* c_d, const unsigned int infinito,
                  unsigned int* minimoDelBloque_d){//reduce3

    extern __shared__ unsigned int sdata[];

    // perform first level of reduction,
    // reading from global memory, writing to shared memory
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x*(blockDim.x*2) + threadIdx.x;
    
    unsigned int dato1= infinito;
    unsigned int dato2= infinito;

    if (p_d[i]) {
        dato1 = c_d[i];
    }

    if (p_d[i+blockDim.x]) {
        dato2 = c_d[i+blockDim.x];
    }
   
    sdata[tid] = min(dato1, dato2);
    __syncthreads();

    // do reduction in shared mem
    for (unsigned int s=blockDim.x/2; s>0; s>>=1) {
        if (tid < s) {
            sdata[tid]= min(sdata[tid], sdata[tid + s]);
        }
        __syncthreads();
    }

    // write result for this block to global mem 
    if (tid == 0) {
        minimoDelBloque_d[blockIdx.x] = sdata[0];
    }
}


////////////////////////////////
__global__ void
kernel_minimizar2(unsigned int* cdi, unsigned int* cdo){//reduce3
    extern __shared__ unsigned int sdata[];

    // perform first level of reduction,
    // reading from global memory, writing to shared memory
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x*(blockDim.x*2) + threadIdx.x;
    
 
    sdata[tid] = min(cdi[i], cdi[i+blockDim.x]);
    __syncthreads();

    // do reduction in shared mem
    for (unsigned int s=blockDim.x/2; s>0; s>>=1) {
        if (tid < s) {
            sdata[tid] = min(sdata[tid], sdata[tid + s]);
        }
        __syncthreads();
    }

    // write result for this block to global mem 
    if (tid == 0) {
        cdo[blockIdx.x] = sdata[0];
    }
}


////////////////////////////////
////////////////////////////////
//////////////     SSSP5 y SSSP6
////////////////////////////////
////////////////////////////////

__global__ void kernel1_SSSP5(const unsigned int* v_d, const unsigned int* a_d, const unsigned int* w_d,
                              const bool* p_d, const bool* f_d, 
                              unsigned int* c_d)
{
    extern __shared__ unsigned int sdata[];

    // access thread id
    const unsigned int tid = threadIdx.x + blockIdx.x * blockDim.x ;
    const unsigned int tx = threadIdx.x;

    sdata[tx] = v_d[tid];
    __syncthreads();
 
    unsigned int i;
    unsigned int tope;
    unsigned int pid;

    if (p_d[tid]) {

        //visitando los predecesores de tid
        i= sdata[tx];

        if(tx==(blockDim.x-1)) tope = v_d[tid+1];
        else tope= sdata[tx+1];

        while (i<tope) {
            pid = a_d[i];

            if (f_d[pid]) {
                c_d[tid] = min(c_d[tid], c_d[pid] + w_d[i]);
            }
            i++;
        }
    }   
}

////////////////////////////////
__global__ void
kernel1_SSSP5_tex( const bool* p_d, const bool* f_d, 
                   unsigned int* c_d)
{
    extern __shared__ unsigned int sdata[];

    // access thread id
    const unsigned int tid = threadIdx.x + blockIdx.x * blockDim.x ;
    const unsigned int tx = threadIdx.x;

    sdata[tx] = tex1Dfetch(textura_v, tid);
    __syncthreads();
 
    unsigned int i;
    unsigned int tope;
    unsigned int pid;

    if (p_d[tid]) {

        //visitando los predecesores de tid
        i = sdata[tx];

        if(tx==(blockDim.x-1)) tope= tex1Dfetch(textura_v, tid+1);
        else tope= sdata[tx+1];

        while (i<tope) {
            pid = tex1Dfetch(textura_a,i);

            if (f_d[pid]) {
                c_d[tid] = min(c_d[tid], c_d[pid] + tex1Dfetch(textura_w, i));
            }
            i++;
        }
    }   
}

////////////////////////////////
__global__ void kernel1_SSSP5_tex_all(unsigned int* c_d)
{
    extern __shared__ unsigned int sdata[];

    // access thread id
    const unsigned int tid = threadIdx.x + blockIdx.x * blockDim.x ;
    const unsigned int tx = threadIdx.x;

    sdata[tx] = tex1Dfetch(textura_v, tid);
    __syncthreads();
 
    unsigned int i;
    unsigned int tope;
    unsigned int pid;

    if ((bool)tex1Dfetch(textura_p,tid).x) {

        //visitando los predecesores de tid
        i= sdata[tx];

        if(tx==(blockDim.x-1)) tope= tex1Dfetch(textura_v, tid+1);
        else tope= sdata[tx+1];

        while(i<tope){
            pid= tex1Dfetch(textura_a,i);

            if ((bool)tex1Dfetch(textura_f, pid).x) {
               c_d[tid] = min(c_d[tid], c_d[pid] + tex1Dfetch(textura_w, i));
            }
            i++;
        }
    }   
}

////////////////////////////////
__global__ void kernel1_SSSP6(const unsigned int* v_d, const unsigned int* a_d, const unsigned int* w_d,
                              const bool* p_d, const bool* f_d, 
                              unsigned int* c_d)
{

    extern __shared__ unsigned int sdata[];

    // access thread id
    const unsigned int tid = threadIdx.x + blockIdx.x * blockDim.x ;
    const unsigned int tx = threadIdx.x;

    sdata[tx] = v_d[tid];

    if (tx == (blockDim.x-1)) {
        sdata[tx+1] = v_d[tid+1];
    }

    __syncthreads();
 
    unsigned int i;
    unsigned int tope;
    unsigned int pid;

    if (p_d[tid]) {
        
        //visitando los predecesores de tid
        i = sdata[tx];
        tope = sdata[tx+1];

        while (i<tope) {
            pid = a_d[i];

            if (f_d[pid]) {
               c_d[tid] = min(c_d[tid], c_d[pid] + w_d[i]);
            }
            i++;
        }
    }  
}

////////////////////////////////
__global__ void
kernel1_SSSP6_tex( const bool* p_d, const bool* f_d, 
                   unsigned int* c_d)
{
    extern __shared__ unsigned int sdata[];

  // access thread id
  const unsigned int tid = threadIdx.x + blockIdx.x * blockDim.x ;
  const unsigned int tx = threadIdx.x;

  sdata[tx] = tex1Dfetch(textura_v, tid);
  if(tx==(blockDim.x-1)) sdata[tx+1]= tex1Dfetch(textura_v, tid+1);
  __syncthreads();
 
  unsigned int i;
  unsigned int tope;
  unsigned int pid;
  if (p_d[tid]) {
        
        //visitando los predecesores de tid
        i= sdata[tx];
        tope= sdata[tx+1];

        while(i<tope){
            pid= tex1Dfetch(textura_a,i);
            if(f_d[pid]){
               c_d[tid] = min(c_d[tid], c_d[pid] + tex1Dfetch(textura_w, i));
            }
            i++;
        }
    }   
}



////////////////////////////////
__global__ void kernel1_SSSP6_tex_all(unsigned int* c_d)
{
    extern __shared__ unsigned int sdata[];

  // access thread id
  const unsigned int tid = threadIdx.x + blockIdx.x * blockDim.x ;
  const unsigned int tx = threadIdx.x;

  sdata[tx] = tex1Dfetch(textura_v, tid);
  if(tx==(blockDim.x-1)) sdata[tx+1]= tex1Dfetch(textura_v, tid+1);
  __syncthreads();
 
  unsigned int i;
  unsigned int tope;
  unsigned int pid;
  if((bool)tex1Dfetch(textura_p,tid).x){
        
        //visitando los predecesores de tid
        i= sdata[tx];
        tope= sdata[tx+1];

        while(i<tope){
            pid= tex1Dfetch(textura_a,i);
            if((bool)tex1Dfetch(textura_f, pid).x){
               c_d[tid] = min(c_d[tid], c_d[pid] + tex1Dfetch(textura_w, i));
            }//if
            i++;
        }//while
     
  }//if   
}


////////////////////////////////
__global__ void
kernel3_tex( bool* p_d, bool* f_d, 
             const unsigned int minimo)
{
  // access thread id
  const unsigned int tid = threadIdx.x + blockIdx.x * blockDim.x ;

    f_d[tid] = false;  

    if (p_d[tid] && (tex1Dfetch(textura_c,tid)==minimo)) {
        f_d[tid] = true;
        p_d[tid] = false;
    }
}


////////////////////////////////
////////////////////////////////
//////////////     SSSP8
////////////////////////////////
////////////////////////////////

__global__ void kernel1_SSSP8(const unsigned int* v_d, const unsigned int* a_d, const unsigned int* w_d,
                              const bool* p_d, const bool* f_d, 
                              unsigned int* c_d)
{
    // access thread id
    const unsigned int tid = threadIdx.x + blockIdx.x * blockDim.x;

    unsigned int i;
    unsigned int tope;
    unsigned int sid;

    if (f_d[tid]) { //tid está en la frontera
        //visitando los sucesores de tid
        i = v_d[tid];
        tope = v_d[tid+1];

        while (i<tope) {
            sid = a_d[i];

            if (p_d[sid]) { //sid está pendiente
                if (c_d[sid] > (c_d[tid] + w_d[i])) {
                    c_d[sid] = c_d[tid] + w_d[i];
                } 
            }
            i++;
        }
    }  
}

__global__ void
kernel1_SSSP8_Atomic( const unsigned int* v_d, const unsigned int* a_d, const unsigned int* w_d,
               const bool* p_d, const bool* f_d, 
               unsigned int* c_d)
{
    // access thread id
    const unsigned int tid = threadIdx.x + blockIdx.x * blockDim.x ;

    unsigned int i;
    unsigned int tope;
    unsigned int sid;

    if (f_d[tid]) { //tid está en la frontera
        //visitando los sucesores de tid
        i = v_d[tid];
        tope = v_d[tid+1];

        while (i<tope) {
            sid = a_d[i];
            if (p_d[sid]) { //sid está pendiente
                atomicMin(&(c_d[sid]), c_d[tid] + w_d[i]);

                //if(SDATA(c_d, sid)>    (SDATA(c_d, tid)+ SDATA(w_d,i))){
                //   SDATA(c_d, sid)=  SDATA(c_d, tid)+ SDATA(w_d,i);
                //}//if 
            }
            i++;
        }
    }
}




#endif // #ifndef _TEMPLATE_KERNEL_H_
