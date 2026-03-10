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
 size_t storage_offset; // for a non view tensor it is just 0 and it is useful in slicing
 int using_mmap;
 int* ref_count;
}Tensor;


Tensor *  createEmptyTensor(size_t * shape , int N){
   if(N<=0) return NULL;
   
   size_t * strides = (size_t *)calloc(N,sizeof(size_t));
   if(!strides) return NULL;
  
   size_t * shapes = (size_t *)calloc(N,sizeof(size_t));
   if(!shapes) {
        free(strides);
        return NULL;
   }
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

   int * ref_count = malloc(sizeof(int));  // it will tell hom many tensors share the same data main tensor and others are views
   if(!ref_count){
       free(shapes);
       free(strides);
       
       if(T->using_mmap == 0){
           free(data);
           free(T);
       }else{
           size_t memsize = size * sizeof(double);
           munmap(data , memsize);
           free(T);
       }
       return NULL;
   }
   *ref_count = 1;
   T->data = data;
   T->shapes = shapes;
   T->strides = strides;
   T->size = size;
   T->ndim = N;
   T->ref_count = ref_count;
   T->storage_offset = 0;
   return T;

}

       
void freeTensor(Tensor * T){
   if(!T) return;
   (*T->ref_count)--;

   if( *(T->ref_count) == 0){ // if T shares the data to multiple tensors do not free the data
 
      if(T->using_mmap){
      
       size_t memsize = T->size * sizeof(double);
       munmap(T->data , memsize);

      }else{
           free(T->data);
      
      }

      free(T->ref_count);
       // since the refernce count is zero we can free the memory hold for reference count
       
   }
   
  
   free(T->shapes);
   free(T->strides);
   free(T);

}   
void printTensorInfo(Tensor *T) {
    if (!T) {
        return;
    }

    printf("Tensor Info:\n");
    printf("  ndim: %d\n", T->ndim);
    printf("  size: %zu\n", T->size);
    printf("  using_mmap: %d\n", T->using_mmap);
    printf("  reference count : %d\n",*T->ref_count);
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
       return NULL;
    }
     // this is a special view based transpose so rather than copying mememory it shares the data between the original and the transpose 
     // by just reversing the shapes and strides we can acchieve a transpose
    Tensor * At = malloc(sizeof(Tensor));
    if(!At){
        return NULL;
    }
    size_t * shapes = calloc(A->ndim , sizeof(size_t));
    if(!shapes){
        free(At);
        return NULL;
    }
    size_t * strides = calloc(A->ndim , sizeof(size_t));
    if(!strides){
        free(shapes);
        free(At);
        return NULL;
    }
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
    At->ref_count = A->ref_count;
    At->storage_offset = A->storage_offset;
    (*At->ref_count)++;
    return At;

}
 
int isNull(Tensor * T){
   if(!T){
      return 1;
   }else {
      return 0;
   }
}

 





size_t getOffset(size_t * strides , size_t * indices , int ndim){

   size_t offset = 0;
      
   for(int i=0; i<ndim; i++)
         offset += indices[i] *  strides[i];
    
   return offset;

}

void printTensorRec(Tensor *T, size_t *indices, int dim)
{
    if (dim == T->ndim - 1) {

        printf("[ ");

        for (size_t i = 0; i < T->shapes[dim]; i++) {
            indices[dim] = i;

            size_t offset = getOffset(T->strides, indices, T->ndim);
            printf("%.2f ", T->data[offset + T->storage_offset]);
        }

        printf("]");

    } else {

        printf("[\n");

        for (size_t i = 0; i < T->shapes[dim]; i++) {

            indices[dim] = i;
            printTensorRec(T, indices, dim + 1);

            printf("\n");
        }

        printf("]");
    }
    fflush(stdout);
}

void printTensor(Tensor *T)
{
    if(!T) return;

    size_t *indices = calloc(T->ndim, sizeof(size_t));
    if(!indices) return;

    printTensorRec(T, indices, 0);

    printf("\n");

    free(indices);
}
         
int isContiguous(Tensor * T){
  if(!T) return 0;
  
  size_t expected = 1; // for a contigous memory strides[N-1]=1; N => ndim
  
  for(int i = T->ndim-1; i>=0; i--){
       if( T->strides[i] != expected ) 
            return 0;
       expected *= T->shapes[i];
  }

  return 1;

}

Tensor * reshape(Tensor * T , size_t * new_shapes , int new_ndim){
    // NOTE : This does not work with not contiguous tensors such as a transpose view for that use reshape_safe that copies the data in a contiguous format
    // and returns the transpose of that 
    if(!T){
       return NULL;
    }
    
    if(!isContiguous(T)){
       return NULL;
    }

    // now let us compute the total size if new_size > old_size 
    size_t old_size = T->size;
    size_t new_size = 1;
    
    for(int i=0; i<new_ndim; i++)
       new_size = new_size * new_shapes[i];

    if(new_size != old_size){
        return NULL;
    }
    
    size_t * new_strides = (size_t *)calloc(new_ndim , sizeof(size_t));
    if(!new_strides){
        return NULL;
    }
    new_strides[new_ndim -1] = 1;
    
    for(int i = new_ndim -2; i>=0; i--)
        new_strides[i] = new_strides[i+1] * new_shapes[i+1];    
    
    size_t * shapes = (size_t *)calloc(new_ndim, sizeof(size_t));

    if(!shapes){
        free(new_strides);
        return NULL;
    }

    for(int i=0; i<new_ndim; i++)
        shapes[i] = new_shapes[i]; // copy all the elements to the shapes array that we own
    
    Tensor * R = malloc(sizeof(Tensor));

    if(!R){
       free(shapes);
       free(new_strides);
       return NULL;
    }

    R->data = T->data; // since this is a view it will refer the same memory
    R->size = new_size;
    R->shapes = shapes;
    R->strides = new_strides;
    (*T->ref_count)++;
    R->ref_count = T->ref_count;
    R->using_mmap = T->using_mmap;
    R->ndim = new_ndim;
    R->storage_offset = T->storage_offset;
    
    return R;

}

void flat_to_multi(size_t flat_index, size_t *shape, int ndim, size_t *index){

    size_t remaining = flat_index;

    for(int k = ndim-1; k >= 0; k--){
        index[k] = remaining % shape[k];
        remaining /= shape[k];
    }
}


Tensor * TensorAdd(Tensor * T1 , Tensor * T2){

  if(isNull(T1) || isNull(T2)) return NULL;
  if(T1->ndim!=T2->ndim) return NULL;
  if(T1->size!=T2->size) return NULL;

  for(int i = 0; i < T1->ndim; i++){
       if(T1->shapes[i] != T2->shapes[i]) 
            return NULL;
  }

  Tensor * res = createEmptyTensor(T1->shapes , T1->ndim);
  if(!res) return NULL;

  if(isContiguous(T1) && isContiguous(T2)){
      // since both tensors are stored contiguous it makes our work easy we can add element wise from the flat array
      for(int i =0; i<T1->size; i++)
         res->data[i] = T1->data[i + T1->storage_offset] + T2->data[i + T2->storage_offset];
      return res;

  }else{
    // here we compute strides and convert the logical index into a physical index since it is non contiguous
       size_t index[T1->ndim];  // this a buffer for storing an index set based on the flat index;
       for(int i =0; i<T1->size; i++){
           flat_to_multi(i , T1->shapes , T1->ndim , index);
           size_t T1_offset = getOffset(T1->strides , index , T1->ndim);
           size_t T2_offset = getOffset(T2->strides , index , T2->ndim);
           res->data[i] = T1->data[T1_offset + T1->storage_offset] + T2->data[T2_offset + T2->storage_offset];
           
           
       }  

  }

  return res;       
}


Tensor * TensorSub(Tensor * T1 , Tensor * T2){

  if(isNull(T1) || isNull(T2)) return NULL;
  if(T1->ndim!=T2->ndim) return NULL;
  if(T1->size!=T2->size) return NULL;

  for(int i = 0; i < T1->ndim; i++){
       if(T1->shapes[i] != T2->shapes[i]) 
            return NULL;
  }

  Tensor * res = createEmptyTensor(T1->shapes , T1->ndim);
  if(!res) return NULL;

  if(isContiguous(T1) && isContiguous(T2)){
      // since both tensors are stored contiguous it makes our work easy we can add element wise from the flat array
      for(int i =0; i<T1->size; i++)
         res->data[i] = T1->data[i + T1->storage_offset] - T2->data[i + T2->storage_offset];
      return res;

  }else{
    // here we compute strides and convert the logical index into a physical index since it is non contiguous
       size_t index[T1->ndim];  // this a buffer for storing an index set based on the flat index;
       for(int i =0; i<T1->size; i++){
           flat_to_multi(i , T1->shapes , T1->ndim , index);
           size_t T1_offset = getOffset(T1->strides , index , T1->ndim);
           size_t T2_offset = getOffset(T2->strides , index , T2->ndim);
           res->data[i] = T1->data[T1_offset + T1->storage_offset] - T2->data[T2_offset + T2->storage_offset];
           
           
       }  

  }

  return res;       
}

Tensor * TensorMul(Tensor * T1 , Tensor * T2){

  if(isNull(T1) || isNull(T2)) return NULL;
  if(T1->ndim!=T2->ndim) return NULL;
  if(T1->size!=T2->size) return NULL;

  for(int i = 0; i < T1->ndim; i++){
       if(T1->shapes[i] != T2->shapes[i]) 
            return NULL;
  }

  Tensor * res = createEmptyTensor(T1->shapes , T1->ndim);
  if(!res) return NULL;

  if(isContiguous(T1) && isContiguous(T2)){
      // since both tensors are stored contiguous it makes our work easy we can add element wise from the flat array
      for(int i =0; i<T1->size; i++)
         res->data[i] = T1->data[i + T1->storage_offset] * T2->data[i + T2->storage_offset];
      return res;

  }else{
    // here we compute strides and convert the logical index into a physical index since it is non contiguous
       size_t index[T1->ndim];  // this a buffer for storing an index set based on the flat index;
       for(int i =0; i<T1->size; i++){
           flat_to_multi(i , T1->shapes , T1->ndim , index);
           size_t T1_offset = getOffset(T1->strides , index , T1->ndim);
           size_t T2_offset = getOffset(T2->strides , index , T2->ndim);
           res->data[i] = T1->data[T1_offset + T1->storage_offset] * T2->data[T2_offset + T2->storage_offset];
           
           
       }  

  }

  return res;
}



Tensor * matMul(Tensor * A , Tensor * B){
      if(isNull(A) || isNull(B) ) return NULL;
      if(A->ndim != 2 || B->ndim != 2) return NULL; // only valid for 2d tensors not valid for n dim where n > 2
     // now check the main condition for the multiplication to go through
      if(A->shapes[1] != B->shapes[0]) return NULL; // if the colmn of A is not equal to row of B then A , B are in compatible
      size_t Row_A = A->shapes[0];
      size_t Col_B = B->shapes[1];
      size_t Inner = A->shapes[1]; // or B->shapes[0];
      size_t shapes[2] = {Row_A , Col_B};// the resultant matrix will always
      Tensor * res = createEmptyTensor(shapes , 2); 
      if(!res) return NULL;
      
      for(size_t r=0; r<Row_A; r++){
           for(size_t c=0; c<Col_B; c++){
               double sum=0;
               for(size_t k=0; k<Inner; k++){
                   
                   size_t A_offset = A->strides[0] * r + A->strides[1] *k; // A[r,k]
                   size_t B_offset = B->strides[0] * k + B->strides[1]*c;  // B[k,c]
                   
                   sum = sum + A->data[A_offset + A->storage_offset]  * B->data[B_offset + B->storage_offset]; // A[r,k] * B[k,c] = C[r,c]
               }
               size_t res_offset = res->strides[0]*r + res->strides[1]*c;
               res->data[res_offset + res->storage_offset] = sum;
                
           }
      }

      return res;
}
    

Tensor * materialize(Tensor *T) {
    // Creates a brand new contiguous copy of T it just converts a non contiguous tensor into a contiguous one or can be used to create copies
    Tensor *C = createEmptyTensor(T->shapes, T->ndim);
    for (size_t i = 0; i < T->size; i++) {
        size_t indices[T->ndim];
        flat_to_multi(i, T->shapes, T->ndim, indices);
        size_t src_offset = getOffset(T->strides, indices, T->ndim);
        C->data[i] = T->data[src_offset + T->storage_offset];
    }
    return C;
}

Tensor * reshape_safe(Tensor *T, size_t *new_shapes, int new_ndim) {
    if (!isContiguous(T)) {
        Tensor *C = materialize(T);
        Tensor *R = reshape(C, new_shapes, new_ndim);
        freeTensor(C);
        return R;
    }
    return reshape(T, new_shapes, new_ndim);
}


Tensor * slice_dim(Tensor *T, int dim, size_t start, size_t end){
    
    if(dim < 0 || dim >= T->ndim) return NULL;
    if(start >= end) return NULL; 
    if(end > T->shapes[dim]) return NULL;
    // dim means which dim are we slicing dim =0 rows , dim =1 colmns dim = 2 blocks
    Tensor *S = malloc(sizeof(Tensor));
    if(!S) return NULL;
    
    S->ndim = T->ndim;
    S->data = T->data;
    S->using_mmap = T->using_mmap;
    S->ref_count = T->ref_count;

    (*S->ref_count)++;

    S->shapes = calloc(S->ndim , sizeof(size_t));
    if(!S->shapes){
        free(S);
        return NULL;
    }
    S->strides = calloc(S->ndim , sizeof(size_t));
    if(!S->strides){
        free(S->shapes);
        free(S);
        return NULL;
    }
    for(int i=0; i<T->ndim; i++){
        S->shapes[i] = T->shapes[i];
        S->strides[i] = T->strides[i];
    }

    S->shapes[dim] = end - start;

    S->storage_offset =
        T->storage_offset + start * T->strides[dim];
    S->size = T->size / T->shapes[dim] * (end - start);
    
    return S;
}

    












