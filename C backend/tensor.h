#ifndef TENSOR_H
#define TENSOR_H

#include <stddef.h>

typedef struct Tensor{
 int ndim;
 size_t * shapes;
 size_t * strides;
 double * data;
 size_t size;
 int using_mmap;
 int is_view;
}Tensor;


Tensor * createEmptyTensor(size_t *shape, int N);
Tensor * Transpose(Tensor * A);
int isNull(Tensor * T);
Tensor * TensorAdd(Tensor * T1 , Tensor * T2);
void freeTensor(Tensor *T);
void printTensor(Tensor *T);
void printTensorInfo(Tensor * T);

#endif
