#include<unistd.h>
#include<stdio.h>
#include<stdlib.h>
#include <sys/mman.h>

typedef struct Tensor{
 int ndim;
 size_t * shapes;
 size_t * strides;
 double * data;
 size_t size;
 int using_mmap;
}Tensor;


Tensor *  createEmptyTensor(size_t * shape , int N){
   if(N<=0) return NULL;
   
   size_t * strides = (size_t *)calloc(N,sizeof(size_t));
   if(!strides) return NULL;
  
   size_t * shapes = (size_t *)calloc(N,sizeof(size_t));
   if(!shapes) return NULL;
   size_t memsize;
   size_t size = 1;
   double * data;
   size_t page_size = sysconf(_SC_PAGESIZE);
   for(int i=0; i<N; i++) 
       shapes[i] = shape[i];
  
 
   strides[N-1] = 1;
   for(int i = N-2; i>=0; i--) 
        strides[i] = strides[i+1] * shapes[i+1];
   
   for(int i=0; i<N; i++)
        size = size * shapes[i];

   memsize = sizeof(double) * size;

   Tensor * T = (Tensor *)malloc(sizeof(Tensor));
   if(!T){
     free(strides);
     free(shapes);
     return NULL
   }

   if(memsize < page_size){ // if the total memory required for this tensor is < pagesize usually 4 kb then use calloc other wise we use virtual memory
         data = (double *)calloc(size, sizeof(double));
         if(!data) {
           free(strides);
           free(shapes);
           free(T);
           return NULL;
         }
       
         T->using_mmap = 0;
   }else{
      
      data = mmap(NULL,
                memsize,
                PROT_READ | PROT_WRITE,
                MAP_PRIVATE | MAP_ANONYMOUS,
                -1,
                0);
 
      if(data == MAP_FAILED){
        free(strides);
        free(shapes);
        free(T);  
        return NULL;
      }
      T->using_mmap = 1;
   }

   
   T->data = data;
   T->shapes = shapes;
   T->strides = strides;
   T->size = size;
   T->ndim = N;
   
   return T;

}

       
void freeTensor(Tensor * T){
   if(!T) return;
   if(T->using_mmap){
      
       size_t memsize = T->size * sizeof(double);
       munmap(T->data , memsize);
   }else{
       free(T->data);
   }

   free(T->shapes);
   free(T->strides);

   free(T);

}   
    
