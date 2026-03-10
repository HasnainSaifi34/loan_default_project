#include <stdio.h>
#include <stdlib.h>
#include <time.h>

/* include your tensor file here */
#include "tensor.c"

void fillTensorSequential(Tensor *T)
{
    if (!T) return;

    size_t index[T->ndim];

    for(size_t i = 0; i < T->size; i++)
    {
        flat_to_multi(i, T->shapes, T->ndim, index);
        size_t offset = getOffset(T->strides, index, T->ndim);
        T->data[offset] = (double)i;
    }
}

void verifyAdd(Tensor *A, Tensor *B, Tensor *C)
{
    size_t index[A->ndim];

    for(size_t i = 0; i < A->size; i++)
    {
        flat_to_multi(i, A->shapes, A->ndim, index);

        size_t a_off = getOffset(A->strides, index, A->ndim);
        size_t b_off = getOffset(B->strides, index, B->ndim);
        size_t c_off = getOffset(C->strides, index, C->ndim);

        double expected = A->data[a_off] + B->data[b_off];

        if(C->data[c_off] != expected)
        {
            printf("ERROR ADD at flat=%zu expected=%f got=%f\n",
                   i, expected, C->data[c_off]);
            exit(1);
        }
    }

    printf("Add verification passed\n");
}

void verifySub(Tensor *A, Tensor *B, Tensor *C)
{
    size_t index[A->ndim];

    for(size_t i = 0; i < A->size; i++)
    {
        flat_to_multi(i, A->shapes, A->ndim, index);

        size_t a_off = getOffset(A->strides, index, A->ndim);
        size_t b_off = getOffset(B->strides, index, B->ndim);
        size_t c_off = getOffset(C->strides, index, C->ndim);

        double expected = A->data[a_off] - B->data[b_off];

        if(C->data[c_off] != expected)
        {
            printf("ERROR SUB at flat=%zu expected=%f got=%f\n",
                   i, expected, C->data[c_off]);
            exit(1);
        }
    }

    printf("Sub verification passed\n");
}

int main(){

    size_t shape[2] = {3,3};

    Tensor *A = createEmptyTensor(shape,2);
    if(!A){
        printf("Tensor creation failed\n");
        return 1;
    }

    // Fill tensor with values 1..9
    for(size_t i = 0; i < A->size; i++){
        A->data[A->storage_offset + i] = i + 1;
    }

    printf("\nOriginal Tensor A\n");
    printTensorInfo(A);
    printTensor(A);

    // Slice rows: A[1:]
    Tensor *B = slice_dim(A,0,1,3);

    printf("\nSlice B = A[1:]\n");
    printTensorInfo(B);
    printTensor(B);

    // Slice columns: A[:,1:]
    Tensor *C = slice_dim(A,1,1,3);

    printf("\nSlice C = A[:,1:]\n");
    printTensorInfo(C);
    printTensor(C);

    // Slice on a slice
    Tensor *D = slice_dim(B,1,1,3);

    printf("\nSlice D = B[:,1:]\n");
    printTensorInfo(D);
    printTensor(D);

    // Clean up
    freeTensor(D);
    freeTensor(C);
    freeTensor(B);
    freeTensor(A);

    return 0;
}