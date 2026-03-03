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
 int is_view;
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
     return NULL;
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
   T->is_view = 0;
   return T;

}

       
void freeTensor(Tensor * T){
   if(!T) return;
   if(!T->is_view){ // if T is not a view then only Free its data because T owns it
 
      if(T->using_mmap){
      
       size_t memsize = T->size * sizeof(double);
       munmap(T->data , memsize);

       }else{
           free(T->data);
       }
   }
   

   free(T->shapes);
   free(T->strides);

   free(T);

}   
void printTensorInfo(Tensor *T) {
    if (!T) {
        perror("Tensor is NULL\n");
        return;
    }

    printf("Tensor Info:\n");
    printf("  ndim: %d\n", T->ndim);
    printf("  size: %zu\n", T->size);
    printf("  using_mmap: %d\n", T->using_mmap);
    printf("  is_view : %d\n",T->is_view);
    printf("  shapes: ");
    for(int i = 0; i < T->ndim; i++)
        printf("%zu ", T->shapes[i]);
    printf("\n");

    printf("  strides: ");
    for(int i = 0; i < T->ndim; i++)
        printf("%zu ", T->strides[i]);
    printf("\n");
    fflush(stdout);
}

Tensor * Transpose(Tensor * A){
    if(!A) return NULL;
    if(A->ndim > 2){
       perror("n-dim should be <= 2 but ndim > 2 \n");
       return NULL;
    }
     // this is a special view based transpose so rather than copying mememory it shares the data between the original and the transpose 
     // by just reversing the shapes and strides we can acchieve a transpose
    Tensor * At = malloc(sizeof(Tensor));
    size_t * shapes = calloc(A->ndim , sizeof(size_t));
    size_t * strides = calloc(A->ndim , sizeof(size_t));

    int N = A->ndim;
    int j =0;
    for(int i = N-1; i>=0; i--){
         strides[j] = A->strides[i];
         shapes[j] = A->shapes[i];
         j++;
    }
    At->strides = strides;
    At->shapes = shapes;
    At->ndim = A->ndim;
    At->size = A->size;
    At->using_mmap = A->using_mmap;
    At->data = A->data;
    At->is_view = 1;

    return At;

}
 
int isNull(Tensor * T){
   if(!T){
      return 1;
   }else {
      return 0;
   }
}

 
Tensor * TensorAdd(Tensor * T1 , Tensor * T2){
  if(isNull(T1) || isNull(T2)){
      perror("NULL pointer error \n");
      return NULL;
  }
  if(!(T1->ndim==T2->ndim)){
      perror("Dimensions dont match \n");
      return NULL;
  }

  for(int i = 0; i < T1->ndim; i++){
       if(T1->shapes[i] != T2->shapes[i]) {
            perror("Size mismatch\n");
            return NULL;
       }
  }

  Tensor * res = createEmptyTensor(T1->shapes , T1->ndim);
  for(int r =0; r<T1->shapes[0]; r++){
       for(int c = 0; c <T1->shapes[1];  c++){
            size_t T1_offset = r*T1->strides[0] + c*T1->strides[1];
            size_t T2_offset = r*T2->strides[0] + c*T2->strides[1];

            size_t Res_offset = r* res->strides[0] + c* res->strides[1];
            res->data[Res_offset] = T1->data[T1_offset] + T2->data[T2_offset];
       }
  };

  return res;

}



void printTensor(Tensor *T) {

    if (!T) {
        printf("Tensor is NULL\n");
        return;
    }

    if (T->ndim == 1) {

        printf("[ ");
        for (size_t i = 0; i < T->shapes[0]; i++) {
            size_t offset = i * T->strides[0];
            printf("%.2f ", T->data[offset]);
        }
        printf("]\n");

    } else if (T->ndim == 2) {

        for (size_t r = 0; r < T->shapes[0]; r++) {

            for (size_t c = 0; c < T->shapes[1]; c++) {

                size_t offset =
                    r * T->strides[0] +
                    c * T->strides[1];

                printf("%lf ", T->data[offset]);
            }

            printf("\n");
        }

    } else {

        printf("printTensor supports only 1D or 2D tensors\n");
    }
    fflush(stdout);
}

size_t getOffset(size_t * strides , size_t * indices , int ndim){

   size_t offset = 0;
      
   for(int i=0; i<ndim; i++)
         offset += indices[i] *  strides[i];
    
   return offset

}


         
int isContiguous(Tensor * T){
  if(!T) return 0;
  
  size_t expected_stride = 1; // for a contigous memory strides[N-1]=1; N => ndim
  
  for(int i = T->ndim-1; i>=0; i--){
       if( T->strides[i] != expected_stride ) 
            return 0;
       expected *= T->shapes[i];
  }

  return 1;

}

Tensor * reshape(Tensor * T , size_t * new_shapes , int new_ndim){
    if(!T){
       perror("\n NULL pointer error \n");
       return NULL;
    }
    
    if(!isContiguous(T)){
       perror("\n to reshape the tensor must be contigous means a non transpose \n");
       return NULL;
    }

    // now let us compute the total size if new_size > old_size 
    size_t old_size = T->size;
    size_t new_size = 1;
    
    for(int i=0; i<new_ndim; i++)
       new_size = new_size * new_shapes[i];

    if(new_size != old_size){
        perror("Size mismatch the new shape has size more than the original shape \n");
        return NULL;
    }
    
    size_t * new_strides = (size_t *)calloc(new_ndim , sizeof(size_t)):
    new_strides[new_ndim -1] = 1;
    
    for(int i = new_ndim -2; i>=0; i--)
        new_strides[i] = new_strides[i+1] * new_shapes[i+1];    
    
    size_t * shapes = (size_t *)calloc(new_ndim, sizeof(size_t));
    for(int i=0; i<new_ndim; i++)
        shapes[i] = new_shapes[i]; // copy all the elements to the shapes array that we own
    
    Tensor * R = malloc(sizeof(Tensor));
    R->data = T->data; // since this is a view it will refer the same memory
    R->size = new_size;
    R->shapes = shapes;
    R->strides = new_strides;
    R->is_view = 1;
    R->using_mmap = T->using_mmap;
    R->ndim = new_ndim;
    
    return R;

}













    












