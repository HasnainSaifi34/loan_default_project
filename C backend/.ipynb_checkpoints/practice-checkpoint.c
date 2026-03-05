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

int main()
{
   printf("\n===== ND PRINT TEST =====\n");

/* -------- 3D Tensor -------- */

size_t shape3[3] = {2,3,4};

Tensor *T3 = createEmptyTensor(shape3,3);

/* fill sequential values */
for(size_t i = 0; i < T3->size; i++)
{
    size_t idx[3];
    flat_to_multi(i, T3->shapes, T3->ndim, idx);

    size_t off = getOffset(T3->strides, idx, T3->ndim);
    T3->data[off] = i + 1;
}

printf("\n3D Tensor (2x3x4):\n");

size_t indices3[3] = {0};
printTensor(T3, indices3, 0);
printf("\n");

printTensorInfo(T3);


/* -------- 4D Tensor -------- */

size_t shape4[4] = {2,2,2,3};

Tensor *T4 = createEmptyTensor(shape4,4);

/* fill sequential values */
for(size_t i = 0; i < T4->size; i++)
{
    size_t idx[4];
    flat_to_multi(i, T4->shapes, T4->ndim, idx);

    size_t off = getOffset(T4->strides, idx, T4->ndim);
    T4->data[off] = i + 1;
}

printf("\n4D Tensor (2x2x2x3):\n");

size_t indices4[4] = {0};
printTensor(T4, indices4, 0);
printf("\n");

printTensorInfo(T4);


/* free tensors */
freeTensor(T3);
freeTensor(T4);
    return 0;
}
